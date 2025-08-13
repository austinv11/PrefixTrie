# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, infer_types=True, initializedcheck=False, nonecheck=False

from libc.stdlib cimport malloc, free
from libc.stddef cimport size_t
from libc.string cimport strcpy, strlen, memcpy, strncmp, memset
from libc.stdint cimport uintptr_t
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as preinc
from cpython.unicode cimport PyUnicode_AsUTF8AndSize, PyUnicode_FromString
from cpython.bytes cimport PyBytes_FromStringAndSize
import cython

# -----------------------------
# Low-level string helpers
# -----------------------------
ctypedef char* Str

cdef Str py_str_to_c_str(str py_str):
    cdef Py_ssize_t n_bytes
    cdef const char* buf = PyUnicode_AsUTF8AndSize(py_str, &n_bytes)
    cdef Str c_str = <Str> malloc(n_bytes + 1)
    if not c_str:
        raise MemoryError("Failed to allocate memory for C string")
    memcpy(c_str, buf, n_bytes)
    c_str[n_bytes] = '\0'
    return c_str

cdef str c_str_to_py_str(const Str c_str):
    if c_str == NULL:
        return None
    return PyUnicode_FromString(c_str)

# -----------------------------
# Trie node & construction
# -----------------------------

ctypedef struct Alphabet:
    size_t size
    int map[256]  # Assuming ASCII, max 256 characters

cdef struct TrieNode:
    int node_id
    char value
    Str collapsed
    size_t collapsed_len
    TrieNode** children
    vector[int]* children_idx
    TrieNode* skip_to
    TrieNode* parent
    Str leaf_value

cdef TrieNode* create_node(const int node_id, const char value, TrieNode* parent, const size_t alphabet_size):
    cdef TrieNode* node = <TrieNode*>malloc(sizeof(TrieNode))
    if not node:
        raise MemoryError("Failed to allocate memory for TrieNode")
    node.node_id = node_id
    node.value = value
    node.collapsed = NULL
    node.skip_to = NULL
    node.children = <TrieNode**> malloc(alphabet_size * sizeof(TrieNode*))  # Allocate space for children pointers
    if not node.children:
        raise MemoryError("Failed to allocate memory for TrieNode children")
    memset(node.children, 0, alphabet_size * sizeof(TrieNode*))  # Initialize to NULL
    node.children_idx = new vector[int]()
    deref(node.children_idx).reserve(4)
    node.parent = parent
    node.leaf_value = NULL
    return node

cdef inline size_t n_children(const TrieNode* n) noexcept nogil:
    return deref(n.children_idx).size()

cdef inline TrieNode* child_at(const TrieNode* n, const size_t i) noexcept nogil:
    cdef int idx = deref(n.children_idx)[i]
    return n.children[idx]

cdef inline void append_child_at_index(TrieNode* p, TrieNode* c, int idx) noexcept nogil:
    p.children[idx] = c
    c.parent = p
    deref(p.children_idx).push_back(idx)

cdef inline bint is_leaf(const TrieNode* node) noexcept nogil:
    return n_children(node) == 0

cdef inline bint has_complete(const TrieNode* node) noexcept nogil:
    return node.leaf_value != NULL

# -----------------------------
# Search result POD
# -----------------------------

cdef struct SearchResult:
    bint found
    Str found_str
    int corrections

cdef inline bint _is_better(const SearchResult* a, const SearchResult* b) noexcept nogil:
    if a.found != b.found:
        return a.found and not b.found
    return a.corrections < b.corrections

# -----------------------------
# Pure C++ cache (opaque to Python)
# -----------------------------

ctypedef unsigned long long Key  # Cache key, bit magic used to hold state

cdef inline Key make_key(const int node_id, const size_t curr_idx, const bint allow_indels) noexcept nogil:
    return ((<unsigned long long> node_id) << 33) | ((<unsigned long long> curr_idx) << 1) | (
        <unsigned long long> (allow_indels & 1))

cdef struct CacheState:
    unordered_map[Key, SearchResult]* data

cdef inline CacheState* cache_new() nogil:
    cdef CacheState* st = <CacheState*> malloc(sizeof(CacheState))
    if st == NULL:
        with gil:
            raise MemoryError("Failed to allocate CacheState")
    st.data = new unordered_map[Key, SearchResult]()
    return st

cdef inline void cache_reserve(CacheState* st, size_t query_len) noexcept nogil:
    # Speed up allocations by pre-reserving a rough guesstimate on initial size
    deref(st.data).reserve(8 * query_len)

cdef inline void cache_free(CacheState* st) noexcept nogil:
    if st != NULL:
        if st.data != NULL:
            del st.data  # delete C++ unordered_map
        free(st)

cdef inline bint cache_contains(const CacheState* st, const Key key) noexcept nogil:
    return deref(st.data).find(key) != deref(st.data).end()

cdef inline SearchResult cache_get(const CacheState* st, const Key key) noexcept nogil:
    cdef unordered_map[Key, SearchResult].iterator it
    cdef SearchResult out
    out.found = False
    out.found_str = NULL
    out.corrections = 0
    it = deref(st.data).find(key)
    if it != deref(st.data).end():
        out = deref(it).second
    return out

cdef inline bint cache_insert_if_better(CacheState* st, Key key, SearchResult incoming) noexcept nogil:
    cdef unordered_map[Key, SearchResult].iterator it
    cdef pair[Key, SearchResult] p
    it = deref(st.data).find(key)
    if it == deref(st.data).end():
        p.first = key
        p.second = incoming
        deref(st.data).insert(p)
        return True
    else:
        if _is_better(&incoming, &deref(it).second):
            deref(it).second = incoming
            return True
        else:
            return False


cdef void _collect_entries(TrieNode* n, vector[Str]* entries) noexcept nogil:
    """Collect pointers to all leaf strings in the trie."""
    if n == NULL:
        return
    if has_complete(n):
        deref(entries).push_back(n.leaf_value)
    cdef int cn = n_children(n)
    cdef int i
    for i in range(cn):
        _collect_entries(child_at(n, i), entries)

# -----------------------------
# Trie object (only search() exposed to Python)
# -----------------------------
@cython.final
@cython.no_gc
cdef class cPrefixTrie:
    cdef TrieNode* root
    cdef int n_entries
    cdef bint allow_indels
    cdef Alphabet alphabet

    def __init__(self, entries: list[str], allow_indels: bool=False):
        # Scan for alphabet to optimize lookups
        cdef bint[256] seen
        cdef int i, j
        cdef Str c_entry
        cdef str entry
        for i in range(256):
            seen[i] = 0
        for entry in entries:
            c_entry = py_str_to_c_str(entry)
            for j in range(<int> strlen(c_entry)):
                seen[<unsigned char> c_entry[j]] = True
            free(c_entry)
        self.alphabet.size = 0
        for i in range(256):
            if seen[i]:
                self.alphabet.map[i] = <int> self.alphabet.size
                self.alphabet.size += 1
            else:
                self.alphabet.map[i] = -1

        self.root = create_node(0, '\0', NULL, <size_t> self.alphabet.size)
        self.n_entries = 0
        self.allow_indels = allow_indels
        cdef int last_id = 1
        for entry in entries:
            last_id = self._insert(entry, last_id)
            self.n_entries += 1
        self._compile(self.root)

    cpdef object make_iter(self):
        """Create an iterator over all entries in the trie."""
        cdef vector[Str] raw_entries
        raw_entries.reserve(self.n_entries)
        with nogil:
            _collect_entries(self.root, &raw_entries)
        cdef size_t i, m = raw_entries.size()
        cdef list entries = [None] * m
        for i in range(m):
            entries[i] = c_str_to_py_str(raw_entries[i])
        return iter(entries)


    cpdef int n_values(self):
        """
        Get the number of entries in the trie.
        :return: The number of entries in the trie.
        """
        return self.n_entries


    cdef int _insert(self, str entry, size_t last_id):
        cdef TrieNode* node = self.root
        cdef char ch
        cdef bint found, inserted_new = False
        cdef TrieNode* last_node = NULL
        cdef TrieNode* new_node = NULL
        cdef Str c_entry = py_str_to_c_str(entry)
        cdef size_t i, k, n = strlen(c_entry)
        cdef int idx, idx_next

        for i in range(n):
            ch = c_entry[i]
            idx = self.alphabet.map[<unsigned char> ch]
            if idx < 0:
                raise ValueError("Character not in alphabet")
            if node.children[idx] != NULL:
                node = node.children[idx]
            else:
                inserted_new = True
                for k in range(n-1, i-1, -1):
                    new_node = create_node(last_id, c_entry[k], NULL, <size_t> self.alphabet.size)
                    last_id += 1
                    if k == n - 1:
                        new_node.leaf_value = <Str> malloc(strlen(c_entry) + 1)
                        if not new_node.leaf_value:
                            raise MemoryError("Failed to allocate memory for leaf value")
                        strcpy(new_node.leaf_value, c_entry)
                    if last_node != NULL:
                        idx_next = self.alphabet.map[<unsigned char> last_node.value]
                        append_child_at_index(new_node, last_node, idx_next)
                    last_node = new_node
                append_child_at_index(node, last_node, idx)
                break

        if not inserted_new and node.leaf_value == NULL:
            node.leaf_value = <Str> malloc(strlen(c_entry) + 1)
            if not node.leaf_value:
                raise MemoryError("Failed to allocate memory for leaf value")
            strcpy(node.leaf_value, c_entry)
        free(c_entry)
        return last_id

    cdef void _compile(self, TrieNode* node):
        cdef size_t i
        cdef TrieNode* child
        cdef size_t clen
        if not node:
            return

        if has_complete(node):
            node.collapsed = <Str>malloc(2)
            if not node.collapsed:
                raise MemoryError("Failed to allocate memory for collapsed value")
            node.collapsed[0] = node.value
            node.collapsed[1] = '\0'
            node.collapsed_len = 1
            node.skip_to = node
            if is_leaf(node):  # It is a leaf, so we are done
                return

        if n_children(node) == 1:
            self._compile(child_at(node, 0))
            # Root (value == '\0') should not prefix its value
            if node.value == '\0':
                node.collapsed = <Str>malloc(strlen(child_at(node, 0).collapsed) + 1)
                if not node.collapsed:
                    raise MemoryError("Failed to allocate memory for collapsed value")
                strcpy(node.collapsed, child_at(node, 0).collapsed)
                node.collapsed_len = child_at(node, 0).collapsed_len
            else:
                clen = child_at(node, 0).collapsed_len
                node.collapsed = <Str>malloc(clen + 2)  # +1 for node.value, +1 for '\0'
                if not node.collapsed:
                    raise MemoryError("Failed to allocate memory for collapsed value")
                node.collapsed[0] = node.value
                strcpy(node.collapsed + 1, child_at(node, 0).collapsed)
                node.collapsed_len = clen + (1 if node.value != '\0' else 0)
            node.skip_to = child_at(node, 0).skip_to
        else:
            for i in range(n_children(node)):
                child = child_at(node, i)
                self._compile(child)
            node.collapsed = <Str>malloc(2)
            if not node.collapsed:
               raise MemoryError("Failed to allocate memory for collapsed value")
            node.collapsed[0] = node.value
            node.collapsed[1] = '\0'
            node.collapsed_len = 1
            node.skip_to = node

    cpdef tuple[str, bint] search(self, str query, int correction_budget=0):
        cdef Str c_query = py_str_to_c_str(query)
        cdef str found_str_py = None
        cdef size_t query_len = strlen(c_query)
        cdef CacheState* st = cache_new()
        cache_reserve(st, query_len)  # Pre-allocate some space
        cdef SearchResult res
        with nogil:
            res = self._search(
                st, self.root, c_query, query_len,
                0, 0, correction_budget, self.allow_indels, False
            )
        if res.found:
            found_str_py = c_str_to_py_str(res.found_str)
        cdef bint exact = res.found and (res.corrections == 0)
        cache_free(st)  # frees all owned SearchResult*
        free(c_query)
        return (found_str_py, exact)

    cdef SearchResult _search(self,
                               CacheState* st,
                               TrieNode* node,
                               Str query,
                               size_t query_len,
                               size_t curr_idx,
                               int curr_corrections,
                               int max_corrections,
                               bint allow_indels,
                               bint exact_only) noexcept nogil:
        # If there is no more remaining corrections, we can get some speed ups by annotating exact_only as true
        exact_only = exact_only or (curr_corrections >= max_corrections)

        cdef SearchResult result
        cdef SearchResult potential
        cdef SearchResult prev
        cdef SearchResult best_result
        cdef bint is_at_query_end = (query_len == curr_idx)
        cdef size_t i
        cdef char query_char
        cdef size_t skip_len
        cdef size_t remaining_chars
        cdef size_t m
        cdef int want
        cdef int idx_child
        cdef int ai
        cdef Str skip_str
        cdef TrieNode* child_node

        result.found = False
        result.corrections = curr_corrections
        result.found_str = NULL

        # Cache check
        cdef Key node_key = make_key(node.node_id, curr_idx, allow_indels)
        if cache_contains(st, node_key):
            prev = cache_get(st, node_key)
            if prev.corrections <= curr_corrections:
                return prev

        # Base cases
        if is_at_query_end and has_complete(node):
            result.found = True
            result.found_str = node.leaf_value
            if not cache_insert_if_better(st, node_key, result):
                return cache_get(st, node_key)
            return result

        # If we reached a complete node but haven't consumed all the query, this is only valid with indels
        if has_complete(node) and not is_at_query_end:
            if allow_indels and curr_corrections < max_corrections:
                # We can match here but need to add corrections for remaining query characters
                remaining_chars = query_len - curr_idx
                if curr_corrections + remaining_chars <= max_corrections:
                    potential.found = True
                    potential.found_str = node.leaf_value
                    potential.corrections = curr_corrections + remaining_chars
                    cache_insert_if_better(st, node_key, potential)
                    result = potential

        if is_at_query_end and (not allow_indels or curr_corrections >= max_corrections):
            if not cache_insert_if_better(st, node_key, result):
                return cache_get(st, node_key)
            return result

        if (not is_at_query_end) and is_leaf(node) and (not allow_indels or curr_corrections >= max_corrections):
            if not cache_insert_if_better(st, node_key, result):
                return cache_get(st, node_key)
            return result

        # Collapsed exact skip
        if node.collapsed is not NULL and node.skip_to is not NULL and node.skip_to != node:
            skip_len = node.collapsed_len
            skip_str = node.collapsed
            if node.value != '\0':  # Root node should not count its value
                skip_len -= 1
                skip_str = node.collapsed + 1
            if skip_len > 0:
                # For exact skip, the collapsed path must match AND we must be able to consume it exactly
                if curr_idx + skip_len <= query_len and strncmp(skip_str, query + curr_idx, skip_len) == 0:
                    # Only do exact skip if we're at the end of query or if we're allowing corrections
                    if curr_idx + skip_len == query_len or allow_indels:
                        potential = self._search(st, node.skip_to, query, query_len, curr_idx + skip_len,
                                                 curr_corrections, max_corrections, allow_indels, exact_only)
                        if potential.found and (exact_only or potential.corrections <= curr_corrections or not allow_indels):
                            return potential
                        # Otherwise cache it and continue exploring
                        cache_insert_if_better(st, make_key(node.skip_to.node_id, curr_idx + skip_len, allow_indels), potential)

        # Exact character match (O(1) via alphabet index)
        if curr_idx < query_len:
            query_char = query[curr_idx]
            ai = self.alphabet.map[<unsigned char> query_char]
            if ai >= 0:
                if node.children[ai] != NULL:
                    potential = self._search(st, node.children[ai], query, query_len, curr_idx + 1,
                                             curr_corrections, max_corrections, allow_indels, exact_only)
                    if potential.found:
                        return potential
                    cache_insert_if_better(st, make_key(node.children[ai].node_id, curr_idx + 1, allow_indels), potential)

        # No budget to correct
        if exact_only or curr_corrections >= max_corrections:
            if not cache_insert_if_better(st, node_key, result):
                return cache_get(st, node_key)
            else:
                return result

        # Try mismatches (iterate only existing children indices)
        if curr_idx < query_len:
            query_char = query[curr_idx]
            want = self.alphabet.map[<unsigned char> query_char]
            m = n_children(node)
            best_result.found = False
            best_result.corrections = max_corrections + 1
            for i in range(m):
                idx_child = deref(node.children_idx)[i]
                if idx_child == want:
                    continue
                child_node = node.children[idx_child]
                potential = self._search(st, child_node, query, query_len, curr_idx + 1,
                                         curr_corrections + 1, max_corrections, allow_indels, exact_only)
                cache_insert_if_better(st, make_key(child_node.node_id, curr_idx + 1, allow_indels), potential)
                if potential.found and potential.corrections < best_result.corrections:
                    best_result = potential

            if best_result.found:
                return best_result


        # Try indels if allowed
        if allow_indels:
            # Insertion: advance in trie (consume trie character) while staying at same query position
            # This simulates inserting a character from the trie into the query
            m = n_children(node)
            for i in range(m):
                idx_child = deref(node.children_idx)[i]
                potential = self._search(st, node.children[idx_child], query, query_len, curr_idx,
                                         curr_corrections + 1, max_corrections, True, exact_only)
                if potential.found:
                    return potential
                cache_insert_if_better(st, make_key(node.children[idx_child].node_id, curr_idx, True), potential)

            # Deletion: advance query index while staying at same trie node
            # This simulates deleting a character from the query
            if curr_idx < query_len:
                potential = self._search(st, node, query, query_len, curr_idx + 1,
                                         curr_corrections + 1, max_corrections, True, exact_only)
                if potential.found:
                    return potential
                cache_insert_if_better(st, make_key(node.node_id, curr_idx + 1, True), potential)

        # Store this node's best result and return
        cache_insert_if_better(st, node_key, result)
        return result

    cdef void _collect_nodes(self, TrieNode* node, list nodes):
        if node == NULL:
            return
        nodes.append(<uintptr_t>node)
        cdef size_t i, m = n_children(node)
        for i in range(m):
            self._collect_nodes(child_at(node, i), nodes)

    cpdef bytes to_bytes(self):
        from struct import pack
        cdef list nodes = []
        self._collect_nodes(self.root, nodes)
        cdef dict node_indices = {}
        cdef Py_ssize_t i, m = len(nodes)
        for i in range(m):
            node_indices[nodes[i]] = i

        cdef bytearray out = bytearray()
        out.extend(pack('<B', self.allow_indels))
        out.extend(pack('<I', self.n_entries))
        out.extend(pack('<I', self.alphabet.size))
        out.extend(pack('<256i', *[self.alphabet.map[i] for i in range(256)]))
        out.extend(pack('<I', m))

        cdef TrieNode* node
        cdef int j, idx_child, child_idx, skip_idx, n_child
        cdef size_t collapsed_len, leaf_len
        for i in range(m):
            node = <TrieNode*> <uintptr_t> nodes[i]
            collapsed_len = node.collapsed_len if node.collapsed != NULL else 0
            leaf_len = strlen(node.leaf_value) if node.leaf_value != NULL else 0
            skip_idx = -1
            if node.skip_to != NULL:
                skip_idx = node_indices.get(<uintptr_t>node.skip_to, -1)
            out.extend(pack('<i', node.node_id))
            out.extend(pack('<b', node.value))
            out.extend(pack('<I', collapsed_len))
            out.extend(pack('<I', leaf_len))
            out.extend(pack('<i', skip_idx))
            n_child = n_children(node)
            out.extend(pack('<I', n_child))
            for j in range(n_child):
                idx_child = deref(node.children_idx)[j]
                child_idx = node_indices[<uintptr_t>node.children[idx_child]]
                out.extend(pack('<i', idx_child))
                out.extend(pack('<i', child_idx))
            if collapsed_len > 0:
                out.extend(PyBytes_FromStringAndSize(node.collapsed, collapsed_len))
            if leaf_len > 0:
                out.extend(PyBytes_FromStringAndSize(node.leaf_value, leaf_len))
        return bytes(out)

    @staticmethod
    def from_bytes(bytes data):
        from struct import unpack_from, calcsize
        cdef cPrefixTrie trie = cPrefixTrie.__new__(cPrefixTrie)
        cdef Py_ssize_t offset = 0
        allow_indels = unpack_from('<B', data, offset)[0]; offset += calcsize('<B')
        n_entries = unpack_from('<I', data, offset)[0]; offset += calcsize('<I')
        alphabet_size = unpack_from('<I', data, offset)[0]; offset += calcsize('<I')
        alphabet_map = unpack_from('<256i', data, offset); offset += calcsize('<256i')
        n_nodes = unpack_from('<I', data, offset)[0]; offset += calcsize('<I')

        trie.allow_indels = bool(allow_indels)
        trie.n_entries = n_entries
        trie.alphabet.size = alphabet_size
        cdef int ai
        for ai in range(256):
            trie.alphabet.map[ai] = alphabet_map[ai]

        cdef list nodes = [0] * n_nodes
        cdef list child_specs = [None] * n_nodes
        cdef list skip_indices = [0] * n_nodes
        cdef TrieNode* node

        cdef int node_id, value, skip_idx, n_child, j, idx_child, child_idx
        cdef size_t collapsed_len, leaf_len
        cdef bytes collapsed_bytes, leaf_bytes
        cdef const char* collapsed_ptr
        cdef const char* leaf_ptr
        cdef list child_pairs
        for ai in range(n_nodes):
            node_id = unpack_from('<i', data, offset)[0]; offset += 4
            value = unpack_from('<b', data, offset)[0]; offset += 1
            collapsed_len = unpack_from('<I', data, offset)[0]; offset += 4
            leaf_len = unpack_from('<I', data, offset)[0]; offset += 4
            skip_idx = unpack_from('<i', data, offset)[0]; offset += 4
            n_child = unpack_from('<I', data, offset)[0]; offset += 4
            child_pairs = []
            for j in range(n_child):
                idx_child = unpack_from('<i', data, offset)[0]; offset += 4
                child_idx = unpack_from('<i', data, offset)[0]; offset += 4
                child_pairs.append((idx_child, child_idx))
            collapsed_bytes = data[offset:offset+collapsed_len]; offset += collapsed_len
            leaf_bytes = data[offset:offset+leaf_len]; offset += leaf_len
            if collapsed_len > 0:
                collapsed_ptr = collapsed_bytes
            else:
                collapsed_ptr = NULL
            if leaf_len > 0:
                leaf_ptr = leaf_bytes
            else:
                leaf_ptr = NULL
            node = create_node(node_id, <char>value, NULL, <size_t>alphabet_size)
            if collapsed_len > 0:
                node.collapsed = <Str>malloc(collapsed_len + 1)
                if node.collapsed == NULL:
                    raise MemoryError("Failed to allocate memory for collapsed value")
                memcpy(node.collapsed, collapsed_ptr, collapsed_len)
                node.collapsed[collapsed_len] = '\0'
                node.collapsed_len = collapsed_len
            else:
                node.collapsed = NULL
                node.collapsed_len = 0
            if leaf_len > 0:
                node.leaf_value = <Str>malloc(leaf_len + 1)
                if node.leaf_value == NULL:
                    raise MemoryError("Failed to allocate memory for leaf value")
                memcpy(node.leaf_value, leaf_ptr, leaf_len)
                node.leaf_value[leaf_len] = '\0'
            else:
                node.leaf_value = NULL
            nodes[ai] = <uintptr_t>node
            child_specs[ai] = child_pairs
            skip_indices[ai] = skip_idx

        for ai in range(n_nodes):
            node = <TrieNode*> <uintptr_t> nodes[ai]
            for idx_child, child_idx in child_specs[ai]:
                append_child_at_index(node, <TrieNode*> <uintptr_t> nodes[child_idx], idx_child)
            skip_idx = skip_indices[ai]
            if skip_idx >= 0:
                node.skip_to = <TrieNode*> <uintptr_t> nodes[skip_idx]
            else:
                node.skip_to = NULL

        trie.root = <TrieNode*> <uintptr_t> nodes[0]
        return trie, bool(allow_indels)

    def __dealloc__(self):
        self._free_node(self.root)

    cdef void _free_node(self, TrieNode* node):
        if not node:
            return
        cdef size_t i, m
        m = n_children(node)
        for i in range(m):
            self._free_node(child_at(node, i))
        if node.children_idx != NULL:
            del node.children_idx  # delete C++ vector
        if node.children != NULL:
            free(node.children)  # free C array of children pointers
        if node.collapsed:
            free(node.collapsed)
        if node.leaf_value:
            free(node.leaf_value)
        free(node)
