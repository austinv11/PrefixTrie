# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, infer_types=True, initializedcheck=False, nonecheck=False

from libc.stdlib cimport malloc, free
# from libc.stddef cimport size_t
from libc.string cimport strcpy, strlen, memcpy, memset
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as preinc
from cpython.unicode cimport PyUnicode_AsUTF8AndSize, PyUnicode_FromString
import cython

# -----------------------------
# Low-level string helpers
# -----------------------------
cdef extern from "simd_compare.h" nogil:
    int simd_strncmp(const char* s1, const char* s2, size_t n)
    size_t simd_strlen(const char* s)
    const char* simd_strchr_any(const char* s, size_t n, const char* needles, size_t num_needles)

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
    # Hot data pointers (first for better cache locality)
    TrieNode** children
    Str leaf_value
    TrieNode* skip_to
    Str collapsed
    vector[int]* children_idx

    # Hot data for search - numeric values
    size_t collapsed_len
    size_t min_remaining
    size_t max_remaining
    int node_id

    # Cold data
    char value
    TrieNode* parent
# -----------------------------
# TrieNode Memory Pool
# -----------------------------
cdef size_t POOL_BLOCK_SIZE = 1024  # Allocate 1024 nodes at a time

@cython.final
@cython.no_gc
cdef class TrieNodePool:
    cdef vector[TrieNode*] blocks
    cdef vector[TrieNode*] free_list
    cdef size_t next_in_block

    def __cinit__(self):
        self.next_in_block = POOL_BLOCK_SIZE  # Start as exhausted to force first allocation

    def __dealloc__(self):
        cdef size_t i
        for i in range(self.blocks.size()):
            free(self.blocks[i])
        self.blocks.clear()
        self.free_list.clear()

    cdef TrieNode* get_node(self) except NULL:
        cdef TrieNode* node
        cdef TrieNode* new_block
        if not self.free_list.empty():
            node = self.free_list.back()
            self.free_list.pop_back()
            return node

        if self.next_in_block >= POOL_BLOCK_SIZE:
            # Current block is exhausted, allocate a new one
            new_block = <TrieNode*> malloc(POOL_BLOCK_SIZE * sizeof(TrieNode))
            if not new_block:
                raise MemoryError("Failed to allocate TrieNode block")
            self.blocks.push_back(new_block)
            self.next_in_block = 0

        node = &self.blocks.back()[self.next_in_block]
        self.next_in_block += 1
        return node

    cdef void return_node(self, TrieNode* node) noexcept:
        # For now, this will only be used if we implement node pruning in mutable tries
        self.free_list.push_back(node)

cdef inline size_t n_children(const TrieNode* n) noexcept nogil:
    return deref(n.children_idx).size()

cdef inline TrieNode* child_at(const TrieNode* n, const size_t i) noexcept nogil:
    cdef int idx = <int> deref(n.children_idx)[i]
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

# New structure for substring search results
cdef struct SubstringSearchResult:
    bint found
    Str found_str
    int corrections
    size_t start_pos  # Starting position in the target string
    size_t end_pos  # Ending position in the target string

cdef inline bint _is_better(const SearchResult* a, const SearchResult* b) noexcept nogil:
    if a.found != b.found:
        return a.found and not b.found
    return a.corrections < b.corrections

cdef inline bint _is_better_substring(const SubstringSearchResult* a, const SubstringSearchResult* b) noexcept nogil:
    if a.found != b.found:
        return a.found and not b.found
    if a.corrections != b.corrections:
        return a.corrections < b.corrections
    # Return true if the new result has a longer length
    return (a.end_pos - a.start_pos) > (b.end_pos - b.start_pos)
# -----------------------------
# Pure C++ cache (opaque to Python)
# -----------------------------

ctypedef unsigned long long Key  # Cache key, bit magic used to hold state

cdef inline Key make_key(const int node_id, const size_t curr_idx, const bint allow_indels) noexcept nogil:
    return ((<unsigned long long> node_id) << 33) | ((<unsigned long long> curr_idx) << 1) | (
        <unsigned long long> (allow_indels & 1))

cdef struct CacheState:
    unordered_map[Key, SearchResult] * data

cdef inline CacheState * cache_new() nogil:
    cdef CacheState * st = <CacheState *> malloc(sizeof(CacheState))
    if st == NULL:
        with gil:
            raise MemoryError("Failed to allocate CacheState")
    st.data = new unordered_map[Key, SearchResult]()
    return st

cdef inline void cache_reserve(CacheState * st, size_t query_len) noexcept nogil:
    # Speed up allocations by pre-reserving a rough guesstimate on initial size
    deref(st.data).reserve(<size_t> (8 * query_len))

cdef inline void cache_free(CacheState * st) noexcept nogil:
    if st != NULL:
        if st.data != NULL:
            del st.data  # delete C++ unordered_map
        free(st)

cdef inline bint cache_contains(const CacheState * st, const Key key) noexcept nogil:
    return deref(st.data).find(key) != deref(st.data).end()

cdef inline SearchResult cache_get(const CacheState * st, const Key key) noexcept nogil:
    cdef unordered_map[Key, SearchResult].iterator it
    cdef SearchResult out
    out.found = False
    out.found_str = NULL
    out.corrections = 0
    it = deref(st.data).find(key)
    if it != deref(st.data).end():
        out = deref(it).second
    return out

# Combine contains and get calls
cdef inline bint cache_try_get(const CacheState * st, Key key, SearchResult* out) noexcept nogil:
    cdef unordered_map[Key, SearchResult].iterator it
    it = deref(st.data).find(key)
    if it != deref(st.data).end():
        out[0] = deref(it).second
        return True
    return False

# Store if better, then return the best result
cdef inline SearchResult cache_store_if_better(CacheState * st, Key key, SearchResult incoming) noexcept nogil:
    cdef unordered_map[Key, SearchResult].iterator it
    cdef pair[Key, SearchResult] p
    it = deref(st.data).find(key)
    if it == deref(st.data).end():
        p.first = key
        p.second = incoming
        deref(st.data).insert(p)
        return incoming
    else:
        if _is_better(&incoming, &deref(it).second):
            deref(it).second = incoming
            return incoming
        else:
            return deref(it).second

# Cache specifically meant for counting

# New cache for counting
cdef inline Key make_count_key(const int node_id, const size_t curr_idx, const int corrections) noexcept nogil:
    # Key composition: 31 bits for node_id, 28 bits for curr_idx, 5 bits for corrections
    return ((<unsigned long long> node_id) << 33) | ((<unsigned long long> curr_idx) << 5) | (<unsigned long long> corrections)

cdef struct CountCacheState:
    unordered_map[Key, bint]* data

cdef inline CountCacheState* count_cache_new() nogil:
    cdef CountCacheState* st = <CountCacheState*> malloc(sizeof(CountCacheState))
    if st == NULL:
        with gil:
            raise MemoryError("Failed to allocate CountCacheState")
    st.data = new unordered_map[Key, bint]()
    return st

cdef inline void count_cache_free(CountCacheState* st) noexcept nogil:
    if st != NULL:
        if st.data != NULL:
            del st.data
        free(st)

cdef inline bint count_cache_is_visited(const CountCacheState* st, Key key) noexcept nogil:
    return deref(st.data).count(key) > 0

cdef inline void count_cache_mark_visited(CountCacheState* st, Key key) noexcept nogil:
    deref(st.data)[key] = True


cdef void _traverse(TrieNode* n, list entries):
    if n is NULL:
        return
    cdef size_t cn = n_children(n)
    cdef size_t i
    if has_complete(n):
        entries.append(c_str_to_py_str(n.leaf_value))
    if cn > 0:
        for i in range(cn):
            _traverse(child_at(n, i), entries)
# -----------------------------
# Trie Iterator
# -----------------------------
@cython.final
@cython.no_gc
cdef class TrieIterator:
    cdef vector[TrieNode*] stack
    cdef TrieNode* root

    def __cinit__(self, cPrefixTrie trie):
        self.root = trie.root
        self.stack.push_back(self.root)

    def __iter__(self):
        return self

    def __next__(self):
        cdef TrieNode* current
        cdef size_t i
        cdef size_t m
        while not self.stack.empty():
            current = self.stack.back()
            self.stack.pop_back()

            # Push children to the stack in reverse order to maintain sorted iteration
            m = n_children(current)
            for i in range(m):
                self.stack.push_back(child_at(current, m - 1 - i))

            if has_complete(current):
                return c_str_to_py_str(current.leaf_value)
        raise StopIteration()

# -----------------------------
# Trie object (only search() exposed to Python)
# -----------------------------
@cython.final
@cython.no_gc
cdef class cPrefixTrie:
    cdef TrieNodePool node_pool
    cdef TrieNode* root
    cdef int n_entries
    cdef bint allow_indels
    cdef bint immutable
    cdef Alphabet alphabet
    cdef size_t max_length
    cdef size_t min_length
    cdef int last_node_id

    def __cinit__(self, *args, **kwargs):
        self.node_pool = TrieNodePool()

    def __init__(self, entries: list[str], allow_indels: bool = False, immutable: bool = True):
        # Initialize alphabet with full 256-character support for mutable tries
        # For immutable tries, we can still optimize by scanning only the entries
        cdef bint[256] seen
        cdef int i, j
        cdef Str c_entry
        cdef str entry

        if immutable:
            # For immutable tries, scan entries to optimize alphabet
            for i in range(256):
                seen[i] = 0
            for entry in entries:
                c_entry = py_str_to_c_str(entry)
                for j in range(<int> simd_strlen(c_entry)):
                    seen[<unsigned char> c_entry[j]] = True
                free(c_entry)
            self.alphabet.size = 0
            for i in range(256):
                if seen[i]:
                    self.alphabet.map[i] = <int> self.alphabet.size
                    self.alphabet.size += 1
                else:
                    self.alphabet.map[i] = -1
        else:
            # For mutable tries, use full alphabet to allow adding any character
            self.alphabet.size = 256
            for i in range(256):
                self.alphabet.map[i] = i

        self.root = self._create_node(0, '\0', NULL, <size_t> self.alphabet.size)
        self.n_entries = 0
        self.allow_indels = allow_indels
        self.immutable = immutable
        self.max_length = 0
        self.min_length = <size_t> -1  # Maximum possible size_t value
        self.last_node_id = 1
        cdef int last_id = 1
        for entry in entries:
            last_id = self._insert(entry, last_id)
            self.n_entries += 1
        self.last_node_id = last_id  # Store the last used ID
        self._compile(self.root)  # Compile the Trie
        self._compute_length_bounds(self.root)  # Compute min/max remaining lengths

    def __iter__(self):
        return TrieIterator(self)

    cpdef int n_values(self):
        """
        Get the number of entries in the trie.
        :return: The number of entries in the trie.
        """
        return self.n_entries

    cpdef int get_min_length(self):
        """
        Get the minimum length of entries in the trie.
        :return: The minimum length of entries in the trie.
        """
        return <int> self.min_length

    cpdef int get_max_length(self):
        """
        Get the maximum length of entries in the trie.
        :return: The maximum length of entries in the trie.
        """
        return <int> self.max_length

    cpdef bint add(self, str entry):
        """
        Add a new entry to the trie (only if mutable).
        :param entry: The string to add
        :return: True if added successfully, False if already exists or trie is immutable
        """
        # Check if entry already exists
        cdef tuple search_result = self.search(entry)
        if search_result[0] is not None:
            return False  # Already exists

        # Add the entry
        self.last_node_id = self._insert(entry, self.last_node_id)
        self.n_entries += 1

        # Recompile the trie to update collapsed paths and bounds
        self._compile(self.root)
        self._compute_length_bounds(self.root)

        return True

    cpdef bint remove(self, str entry):
        """
        Remove an entry from the trie (only if mutable).
        :param entry: The string to remove
        :return: True if removed successfully, False if not found or trie is immutable
        """
        # Check if entry exists
        cdef tuple search_result = self.search(entry)
        if search_result[0] is None:
            return False  # Not found

        # Remove the entry by clearing its leaf value
        cdef bint removed = self._remove_entry(entry)
        if removed:
            self.n_entries -= 1
            # Recompile the trie to update collapsed paths and bounds
            self._compile(self.root)
            self._compute_length_bounds(self.root)

        return removed

    cdef bint _remove_entry(self, str entry):
        """
        Internal method to remove an entry from the trie.
        :param entry: The string to remove
        :return: True if removed successfully
        """
        cdef TrieNode* node = self.root
        cdef char ch
        cdef Str c_entry = py_str_to_c_str(entry)
        cdef size_t i, n = simd_strlen(c_entry)
        cdef int idx

        try:
            # Navigate to the node containing the entry
            for i in range(n):
                ch = c_entry[i]
                idx = self.alphabet.map[<unsigned char> ch]
                if idx < 0 or node.children[idx] == NULL:
                    return False  # Path doesn't exist
                node = node.children[idx]

            # Check if this node has the leaf value we want to remove
            if node.leaf_value == NULL:
                return False  # No leaf value at this node

            # Clear the leaf value
            free(node.leaf_value)
            node.leaf_value = NULL

            return True
        finally:
            free(c_entry)

    cpdef bint is_immutable(self):
        """
        Check if the trie is immutable.
        :return: True if immutable, False if mutable
        """
        return self.immutable

    cdef TrieNode* _create_node(self, const int node_id, const char value, TrieNode* parent,
                                 const size_t alphabet_size):
        cdef TrieNode* node = self.node_pool.get_node()
        # The node from the pool might have old data, so we MUST initialize it.
        node.node_id = node_id
        node.value = value
        node.collapsed = NULL
        node.skip_to = NULL
        node.children = <TrieNode**> malloc(alphabet_size * sizeof(TrieNode*))
        if not node.children:
            raise MemoryError("Failed to allocate memory for TrieNode children")
        memset(node.children, 0, alphabet_size * sizeof(TrieNode*))
        node.children_idx = new vector[int]()
        deref(node.children_idx).reserve(4)
        node.parent = parent
        node.leaf_value = NULL
        node.collapsed_len = 0
        node.min_remaining = 0
        node.max_remaining = 0
        return node

    cdef int _insert(self, str entry, size_t last_id):
        cdef TrieNode* node = self.root
        cdef char ch
        cdef bint found, inserted_new = False
        cdef TrieNode* last_node = NULL
        cdef TrieNode* new_node = NULL
        cdef Str c_entry = py_str_to_c_str(entry)
        cdef size_t i, k, n = simd_strlen(c_entry)
        cdef int idx, idx_next

        if n > self.max_length:
            self.max_length = n
        if n < self.min_length:
            self.min_length = n

        for i in range(n):
            ch = c_entry[i]
            idx = self.alphabet.map[<unsigned char> ch]
            if idx < 0:
                raise ValueError("Character not in alphabet")
            if node.children[idx] != NULL:
                node = node.children[idx]
            else:
                inserted_new = True
                for k in range(n - 1, i - 1, -1):
                    new_node = self._create_node(last_id, c_entry[k], NULL, <size_t> self.alphabet.size)
                    last_id += 1
                    if k == n - 1:
                        new_node.leaf_value = <Str> malloc(simd_strlen(c_entry) + 1)
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
            node.leaf_value = <Str> malloc(simd_strlen(c_entry) + 1)
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

        # Prevent memory leak by freeing existing collapsed value before re-assigning
        if node.collapsed:
            free(node.collapsed)
            node.collapsed = NULL

        if has_complete(node):
            node.collapsed = <Str> malloc(2)
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
                node.collapsed = <Str> malloc(simd_strlen(child_at(node, 0).collapsed) + 1)
                if not node.collapsed:
                    raise MemoryError("Failed to allocate memory for collapsed value")
                strcpy(node.collapsed, child_at(node, 0).collapsed)
                node.collapsed_len = child_at(node, 0).collapsed_len
            else:
                clen = child_at(node, 0).collapsed_len
                node.collapsed = <Str> malloc(clen + 2)  # +1 for node.value, +1 for '\0'
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
            node.collapsed = <Str> malloc(2)
            if not node.collapsed:
                raise MemoryError("Failed to allocate memory for collapsed value")
            node.collapsed[0] = node.value
            node.collapsed[1] = '\0'
            node.collapsed_len = 1
            node.skip_to = node

    cdef pair[size_t, size_t] _compute_length_bounds(self, TrieNode* node) noexcept nogil:
        cdef size_t m = n_children(node)
        cdef size_t i
        cdef pair[size_t, size_t] child_bounds
        cdef size_t min_child, max_child
        if m == 0:
            node.min_remaining = 0
            node.max_remaining = 0
            child_bounds.first = 0
            child_bounds.second = 0
            return child_bounds
        min_child = (<size_t> -1)
        max_child = 0
        for i in range(m):
            child_bounds = self._compute_length_bounds(child_at(node, i))
            if child_bounds.first < min_child:
                min_child = child_bounds.first
            if child_bounds.second > max_child:
                max_child = child_bounds.second
        if has_complete(node) and (min_child + 1) > 0:
            node.min_remaining = 0
        else:
            node.min_remaining = min_child + 1
        node.max_remaining = max_child + 1
        child_bounds.first = node.min_remaining
        child_bounds.second = node.max_remaining
        return child_bounds

    cpdef tuple[str, int] search(self, str query, int correction_budget=0):
        """
        Search for a query in the trie, allowing for a specified number of corrections.
        :param query: The query string to search for.
        :param correction_budget: The maximum number of corrections allowed.
        :return: A tuple containing the found string and the number of corrections,
                 or (None, -1) if no match is found.
        """
        cdef Str c_query = py_str_to_c_str(query)
        cdef str found_str_py = None
        cdef size_t query_len = simd_strlen(c_query)
        cdef CacheState * st = cache_new()
        cache_reserve(st, query_len)  # Pre-allocate some space
        cdef SearchResult res
        with nogil:
            res = self._search(
                st, self.root, c_query, query_len,
                0, 0, correction_budget, self.allow_indels, False
            )
        cache_free(st)
        free(c_query)
        if res.found:
            found_str_py = c_str_to_py_str(res.found_str)
            return found_str_py, res.corrections
        return None, -1

    cpdef tuple[str, int, int, int] search_substring(self, str target_string, int correction_budget=0):
        """
        Search for fuzzy substring matches of trie entries within a target string.

        :param target_string: The string to search within
        :param correction_budget: Maximum number of edits allowed
        :return: Tuple of (found_string, corrections, start_pos, end_pos) or (None, -1, -1, -1)
        """
        cdef Str c_target = py_str_to_c_str(target_string)
        cdef str found_str_py = None
        cdef size_t target_len = simd_strlen(c_target)
        cdef SubstringSearchResult best_result

        # Check if the length of the target string makes it impossible to find a match
        if target_len + correction_budget < self.min_length:
            free(c_target)
            return (None, -1, -1, -1)

        with nogil:
            best_result = self._search_substring_internal(c_target, target_len, correction_budget)

        free(c_target)
        # Convert result to Python types
        if best_result.found:
            found_str_py = c_str_to_py_str(best_result.found_str)
            return found_str_py, best_result.corrections, best_result.start_pos, best_result.end_pos
        else:
            return None, -1, -1, -1

    cpdef tuple[str, int, int] longest_prefix_match(self, str target_string, int min_match_length=1):
        """
        Scan the target string for the longest prefix substring present in the trie.
        Note that this is an exact match search and does not require a complete match.
        However, if there is an incomplete match and it is ambiguous what the next character is,
        the search will return no match.

        Example: Given a trie with "app" and "apple", searching "xyzappl" will return "apple" as the longest prefix match,
        searching "xyzapp" will return "app", but searching "xyzap" will return no match since it is ambiguous.

        :param target_string: The string to search within.
        :param min_match_length: Minimum length of the match to be considered valid.
        :return: Tuple of (found_string, target_start_pos, match_len) or (None, -1, -1) if no match is found.
        """
        cdef Str c_target = py_str_to_c_str(target_string)
        cdef str found_str_py = None
        cdef size_t target_len = simd_strlen(c_target)
        cdef size_t min_match_len_c = <size_t> min_match_length
        if target_len < min_match_len_c:  # Only check if target is too short for min_match_length
            free(c_target)
            return None, -1, -1
        cdef SubstringSearchResult best_result

        with nogil:
            best_result = self._longest_prefix_search(c_target, target_len, min_match_len_c)

        free(c_target)
        # Convert result to Python types
        if best_result.found:
            found_str_py = c_str_to_py_str(best_result.found_str)
            free(best_result.found_str)
            return found_str_py, best_result.start_pos, best_result.end_pos - best_result.start_pos
        else:
            return None, -1, -1

    cpdef int search_count(self, str query, int correction_budget=0):
        """
        Count the number of entries that match a query within a given correction budget.

        :param query: The query string to search for.
        :param correction_budget: The maximum number of corrections allowed.
        :return: The number of matching entries found.
        """
        cdef Str c_query = py_str_to_c_str(query)
        cdef size_t query_len = simd_strlen(c_query)
        cdef unordered_set[Str] found_entries
        cdef CountCacheState* st = count_cache_new()

        # Check if any match is possible
        if <int>self.min_length > <int>query_len + correction_budget and <int>self.max_length < <int>query_len - correction_budget:
            free(c_query)
            count_cache_free(st)
            return 0

        with nogil:
            self._search_count_recursive(
                st, self.root, c_query, query_len, 0, 0, correction_budget, self.allow_indels, &found_entries
            )

        count_cache_free(st)
        free(c_query)
        return found_entries.size()

    cdef SearchResult _search(self,
                              CacheState * st,
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
        cdef size_t len_diff
        cdef size_t m
        cdef int want
        cdef int idx_child
        cdef int ai
        cdef Str skip_str
        cdef TrieNode* child_node
        cdef size_t remaining_query_len
        cdef size_t min_possible_edits
        cdef size_t remaining_query_len_after_sub
        cdef size_t child_len_diff

        result.found = False
        result.corrections = curr_corrections
        result.found_str = NULL

        # Cache check
        cdef Key node_key = make_key(node.node_id, curr_idx, allow_indels)
        if cache_try_get(st, node_key, &prev):
            if prev.corrections <= curr_corrections:
                return prev

        # Base cases
        if is_at_query_end and has_complete(node):
            result.found = True
            result.found_str = node.leaf_value
            return cache_store_if_better(st, node_key, result)

        # If we reached a complete node but haven't consumed all the query, this is only valid with indels
        if has_complete(node) and not is_at_query_end:
            if allow_indels and curr_corrections < max_corrections:
                # We can match here but need to add corrections for remaining query characters
                remaining_chars = query_len - curr_idx
                if curr_corrections + <int> remaining_chars <= max_corrections:
                    potential.found = True
                    potential.found_str = node.leaf_value
                    potential.corrections = curr_corrections + <int> remaining_chars
                    result = cache_store_if_better(st, node_key, potential)

        if is_at_query_end and (not allow_indels or curr_corrections >= max_corrections):
            return cache_store_if_better(st, node_key, result)

        if (not is_at_query_end) and is_leaf(node) and (not allow_indels or curr_corrections >= max_corrections):
            return cache_store_if_better(st, node_key, result)

        # Prune if length difference alone exceeds remaining budget
        remaining_chars = query_len - curr_idx
        if remaining_chars < node.min_remaining:
            len_diff = node.min_remaining - remaining_chars
        elif remaining_chars > node.max_remaining:
            len_diff = remaining_chars - node.max_remaining
        else:
            len_diff = 0
        if curr_corrections + <int> len_diff > max_corrections:
            return cache_store_if_better(st, node_key, result)

        # Collapsed exact skip
        if node.collapsed is not NULL and node.skip_to is not NULL and node.skip_to != node:
            skip_len = node.collapsed_len
            skip_str = node.collapsed
            if node.value != '\0':  # Root node should not count its value
                skip_len -= 1
                skip_str = node.collapsed + 1
            if skip_len > 0:
                # For exact skip, the collapsed path must match AND we must be able to consume it exactly
                if curr_idx + skip_len <= query_len and simd_strncmp(skip_str, query + curr_idx, skip_len) == 0:
                    # Only do exact skip if we're at the end of query or if we're allowing corrections
                    if curr_idx + skip_len == query_len or allow_indels:
                        potential = self._search(st, node.skip_to, query, query_len, curr_idx + skip_len,
                                                 curr_corrections, max_corrections, allow_indels, exact_only)
                        if potential.found and (
                                exact_only or potential.corrections <= curr_corrections or not allow_indels):
                            return potential
                        # Otherwise cache it and continue exploring
                        cache_store_if_better(st, make_key(node.skip_to.node_id, curr_idx + skip_len, allow_indels),
                                              potential)

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
                    cache_store_if_better(st, make_key(node.children[ai].node_id, curr_idx + 1, allow_indels),
                                          potential)

        # No budget to correct
        if exact_only or curr_corrections >= max_corrections:
            return cache_store_if_better(st, node_key, result)

        # Try mismatches (iterate only existing children indices)
        if curr_idx < query_len:
            query_char = query[curr_idx]
            want = self.alphabet.map[<unsigned char> query_char]
            m = n_children(node)
            best_result.found = False
            best_result.corrections = max_corrections + 1
            remaining_query_len_after_sub = query_len - (curr_idx + 1)

            for i in range(m):
                idx_child = <int> deref(node.children_idx)[i]
                if idx_child == want:
                    continue
                child_node = node.children[idx_child]

                # Pruning check: if this substitution path can't possibly be better than our best result so far, skip it.
                child_len_diff = 0
                if remaining_query_len_after_sub < child_node.min_remaining:
                    child_len_diff = child_node.min_remaining - remaining_query_len_after_sub
                elif remaining_query_len_after_sub > child_node.max_remaining:
                    child_len_diff = remaining_query_len_after_sub - child_node.max_remaining

                if (curr_corrections + 1 + <int>child_len_diff) >= best_result.corrections:
                    continue

                potential = self._search(st, child_node, query, query_len, curr_idx + 1,
                                         curr_corrections + 1, max_corrections, allow_indels, exact_only)
                cache_store_if_better(st, make_key(child_node.node_id, curr_idx + 1, allow_indels), potential)
                if potential.found:
                    if potential.corrections == curr_corrections + 1:
                        return potential  # Found an exact match with one correction
                    if potential.corrections < best_result.corrections:
                        best_result = potential

            if best_result.found:
                return best_result

        # Try indels if allowed
        if allow_indels:
            # Insertion: advance in trie (consume trie character) while staying at same query position
            # This simulates inserting a character from the trie into the query
            m = n_children(node)
            for i in range(m):
                idx_child = <int> deref(node.children_idx)[i]
                child_node = node.children[idx_child]

                # Pruning check for insertion path
                len_diff = 0
                remaining_chars = query_len - curr_idx
                if remaining_chars < child_node.min_remaining:
                    len_diff = child_node.min_remaining - remaining_chars
                elif remaining_chars > child_node.max_remaining:
                    len_diff = remaining_chars - child_node.max_remaining

                if (curr_corrections + 1 + <int>len_diff) > max_corrections:
                    continue

                potential = self._search(st, child_node, query, query_len, curr_idx,
                                         curr_corrections + 1, max_corrections, True, exact_only)
                if potential.found:
                    return potential
                cache_store_if_better(st, make_key(child_node.node_id, curr_idx, True), potential)

            # Deletion: advance query index while staying at same trie node
            # This simulates deleting a character from the query
            if curr_idx < query_len:
                # Pruning check for deletion path
                len_diff = 0
                remaining_chars = query_len - (curr_idx + 1)
                if remaining_chars < node.min_remaining:
                    len_diff = node.min_remaining - remaining_chars
                elif remaining_chars > node.max_remaining:
                    len_diff = remaining_chars - node.max_remaining

                if (curr_corrections + 1 + <int>len_diff) <= max_corrections:
                    potential = self._search(st, node, query, query_len, curr_idx + 1,
                                             curr_corrections + 1, max_corrections, True, exact_only)
                    if potential.found:
                        return potential
                    cache_store_if_better(st, make_key(node.node_id, curr_idx + 1, True), potential)

        # Store this node's best result and return
        return cache_store_if_better(st, node_key, result)

    cdef SubstringSearchResult _search_substring_internal(self, Str c_target, size_t target_len,
                                                          int correction_budget) noexcept nogil:
        cdef SubstringSearchResult best_result
        cdef SubstringSearchResult current_result
        cdef size_t start_pos
        cdef CacheState * st

        best_result.found = False
        best_result.found_str = NULL
        best_result.corrections = correction_budget + 1
        best_result.start_pos = 0
        best_result.end_pos = 0

        for start_pos in range(target_len):
            if target_len - start_pos + correction_budget < self.min_length:
                break

            st = cache_new()
            cache_reserve(st, target_len - start_pos)

            current_result.found = False
            current_result.corrections = correction_budget + 1

            self._search_substring_recursive(
                st, self.root, c_target, target_len, start_pos, start_pos, 0, correction_budget, &current_result
            )

            cache_free(st)

            if current_result.found:
                if _is_better_substring(&current_result, &best_result):
                    best_result = current_result
                    if best_result.corrections == 0:
                        if (best_result.end_pos - best_result.start_pos) >= self.max_length:
                            break
        return best_result

    cdef void _search_substring_recursive(self,
                                          CacheState * st,
                                          TrieNode* node,
                                          Str target,
                                          size_t target_len,
                                          size_t start_pos,
                                          size_t curr_idx,
                                          int curr_corrections,
                                          int max_corrections,
                                          SubstringSearchResult* best_result_for_start) noexcept nogil:
        cdef Key node_key
        cdef SearchResult prev
        cdef SubstringSearchResult potential_result
        cdef size_t i, m
        cdef int ai, idx_child
        cdef TrieNode* child_node
        cdef char query_char
        cdef SearchResult res_to_cache

        node_key = make_key(node.node_id, curr_idx - start_pos, True)
        if cache_try_get(st, node_key, &prev):
            if prev.corrections <= curr_corrections:
                return

        if has_complete(node):
            potential_result.found = True
            potential_result.found_str = node.leaf_value
            potential_result.corrections = curr_corrections
            potential_result.start_pos = start_pos
            potential_result.end_pos = curr_idx

            if _is_better_substring(&potential_result, best_result_for_start):
                best_result_for_start[0] = potential_result

        if curr_idx >= target_len:
            return

        # Only return early if we found an exact match that's already the maximum possible length
        if best_result_for_start.found and best_result_for_start.corrections == 0:
            if (best_result_for_start.end_pos - best_result_for_start.start_pos) >= self.max_length:
                return

        # Exact match
        query_char = target[curr_idx]
        ai = self.alphabet.map[<unsigned char> query_char]
        if ai >= 0 and node.children[ai] != NULL:
            self._search_substring_recursive(st, node.children[ai], target, target_len, start_pos, curr_idx + 1,
                                             curr_corrections, max_corrections, best_result_for_start)
        if curr_corrections >= max_corrections:
            res_to_cache.found = False
            res_to_cache.corrections = curr_corrections
            res_to_cache.found_str = NULL
            cache_store_if_better(st, node_key, res_to_cache)
            return

        # Mismatch
        m = n_children(node)
        for i in range(m):
            idx_child = <int> deref(node.children_idx)[i]
            if idx_child == ai:
                continue
            child_node = node.children[idx_child]
            self._search_substring_recursive(st, child_node, target, target_len, start_pos, curr_idx + 1,
                                             curr_corrections + 1, max_corrections, best_result_for_start)

        # Indels
        if self.allow_indels:
            # Insertion
            for i in range(m):
                child_node = child_at(node, i)
                self._search_substring_recursive(st, child_node, target, target_len, start_pos, curr_idx,
                                                 curr_corrections + 1, max_corrections, best_result_for_start)

            # Deletion
            self._search_substring_recursive(st, node, target, target_len, start_pos, curr_idx + 1,
                                             curr_corrections + 1, max_corrections, best_result_for_start)

        res_to_cache.found = False
        res_to_cache.corrections = curr_corrections
        res_to_cache.found_str = NULL
        cache_store_if_better(st, node_key, res_to_cache)

    cdef SubstringSearchResult _longest_prefix_search(self, Str target, size_t target_len,
                                                      size_t min_match_len) noexcept nogil:
        cdef SubstringSearchResult best_result
        cdef TrieNode* node
        cdef size_t start_pos
        cdef size_t curr_idx
        cdef int ai
        cdef Str matched_str
        cdef size_t match_len
        cdef size_t best_match_len = 0
        cdef size_t i

        # Pre-fill best_result
        best_result.found = False
        best_result.found_str = NULL
        best_result.corrections = 0
        best_result.start_pos = 0
        best_result.end_pos = 0

        # Optimization: Collect all possible starting characters from the root's children
        cdef size_t num_root_children = n_children(self.root)
        if num_root_children == 0:
            return best_result # Empty trie

        cdef char* root_children_chars = <char*> malloc(num_root_children * sizeof(char))
        if not root_children_chars:
            # A robust implementation might fall back to a simple loop here.
            # For now, we assume malloc succeeds and return no match if it fails.
            return best_result

        for i in range(num_root_children):
            root_children_chars[i] = child_at(self.root, i).value

        cdef const char* p = target
        cdef const char* end = target + target_len

        while p < end:
            # Use SIMD to quickly find the next character that is a child of the root
            p = simd_strchr_any(p, end - p, root_children_chars, num_root_children)

            if p == NULL or (end - p) < min_match_len:
                break  # No more potential matches

            start_pos = p - target
            node = self.root
            curr_idx = start_pos

            # Walk down the trie as far as possible
            while curr_idx < target_len:
                ai = self.alphabet.map[<unsigned char> target[curr_idx]]
                if ai < 0 or node.children[ai] == NULL:
                    break
                node = node.children[ai]
                curr_idx += 1

                # Check if this node represents a complete entry
                if has_complete(node):
                    match_len = curr_idx - start_pos
                    # Check if this is a valid match and better than our current best
                    if match_len >= min_match_len and match_len > best_match_len:
                        best_match_len = match_len
                        best_start_pos = start_pos

                        # Free previous result if it exists
                        if best_result.found:
                            free(best_result.found_str)

                        # Allocate and copy the matched substring
                        matched_str = <Str> malloc(match_len + 1)
                        if matched_str == NULL:
                            # Memory allocation failed, return current best or nothing
                            break
                        memcpy(matched_str, target + start_pos, match_len)
                        matched_str[match_len] = '\0'

                        best_result.found = True
                        best_result.found_str = matched_str
                        best_result.start_pos = start_pos
                        best_result.end_pos = start_pos + match_len
            p += 1  # Continue searching from the next character

        free(root_children_chars)
        return best_result

    cdef void _search_count_recursive(self,
                                      CountCacheState * st,
                                      TrieNode * node,
                                      Str query,
                                      size_t query_len,
                                      size_t curr_idx,
                                      int curr_corrections,
                                      int max_corrections,
                                      bint allow_indels,
                                      unordered_set[Str]* found_entries) noexcept nogil:
        cdef Key node_key
        cdef size_t i, m, len_diff
        cdef int ai, idx_child
        cdef TrieNode * child_node
        cdef char query_char
        cdef size_t remaining_query_len

        if curr_corrections > max_corrections:
            return

        node_key = make_count_key(node.node_id, curr_idx, curr_corrections)
        if count_cache_is_visited(st, node_key):
            return
        count_cache_mark_visited(st, node_key)

        # Base case: if the query is fully consumed
        if curr_idx == query_len:
            # If current node is a match, add it
            if has_complete(node):
                found_entries.insert(node.leaf_value)

            # We can still match longer words by incurring insertion costs
            if allow_indels:
                m = n_children(node)
                for i in range(m):
                    child_node = child_at(node, i)
                    self._search_count_recursive(st, child_node, query, query_len, curr_idx, curr_corrections + 1,
                                                 max_corrections, allow_indels, found_entries)
            return

        # Pruning based on length difference
        remaining_query_len = query_len - curr_idx
        if remaining_query_len < node.min_remaining:
            len_diff = node.min_remaining - remaining_query_len
        elif remaining_query_len > node.max_remaining:
            len_diff = remaining_query_len - node.max_remaining
        else:
            len_diff = 0
        if curr_corrections + <int>len_diff > max_corrections:
            return

        # Handle match at the current node (deletion of remaining query chars)
        if has_complete(node):
            if allow_indels:
                if curr_corrections + <int>remaining_query_len <= max_corrections:
                    found_entries.insert(node.leaf_value)

        # --- Recursive Steps ---
        # 1. Exact Match
        query_char = query[curr_idx]
        ai = self.alphabet.map[<unsigned char> query_char]
        if ai >= 0 and node.children[ai] is not NULL:
            self._search_count_recursive(st, node.children[ai], query, query_len, curr_idx + 1, curr_corrections,
                                         max_corrections, allow_indels, found_entries)

        # Budget check for further corrections
        if curr_corrections >= max_corrections:
            return

        # 2. Substitution
        m = n_children(node)
        for i in range(m):
            idx_child = <int> deref(node.children_idx)[i]
            if idx_child == ai:
                continue
            child_node = node.children[idx_child]
            self._search_count_recursive(st, child_node, query, query_len, curr_idx + 1, curr_corrections + 1,
                                         max_corrections, allow_indels, found_entries)

        if allow_indels:
            # 3. Deletion (from query)
            self._search_count_recursive(st, node, query, query_len, curr_idx + 1, curr_corrections + 1,
                                         max_corrections, allow_indels, found_entries)

            # 4. Insertion (into query)
            for i in range(m):
                child_node = child_at(node, i)
                self._search_count_recursive(st, child_node, query, query_len, curr_idx, curr_corrections + 1,
                                             max_corrections, allow_indels, found_entries)

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
        # The TrieNode struct itself is not freed, as its memory is managed by the TrieNodePool.
        # The pool will be deallocated when the cPrefixTrie object is deallocated.
