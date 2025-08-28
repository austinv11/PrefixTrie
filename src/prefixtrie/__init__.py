from ._impl import cPrefixTrie
import pickle
import multiprocessing.shared_memory as shm
import weakref
import atexit

try:
    from ._version import __version__
except ImportError:
    # Fallback version if _version.py doesn't exist yet
    __version__ = "unknown"


# Global registry to track shared memory blocks for cleanup
_shared_memory_registry = weakref.WeakSet()
_cleanup_registered = False


def _cleanup_shared_memory():
    """Clean up any remaining shared memory blocks on exit"""
    for shm_block in list(_shared_memory_registry):
        try:
            shm_block.unlink()
        except:
            pass  # Ignore errors during cleanup


class PrefixTrie:
    """
    Thin wrapper around the cPrefixTrie class to provide a Python interface.
    """

    __slots__ = ("_trie", "allow_indels", "_entries", "_shared_memory", "_is_shared_owner", "_exact_set")

    def __init__(self, entries: list[str], allow_indels: bool=False, shared_memory_name: str=None):
        """
        Initialize the PrefixTrie with the given arguments.
        :param entries: List of strings to be added to the trie.
        :param allow_indels: If True, allows insertions and deletions in the trie
        :param shared_memory_name: If provided, load from existing shared memory block
        """
        global _cleanup_registered

        if shared_memory_name:
            # Load from shared memory
            self._load_from_shared_memory(shared_memory_name)
        else:
            # Normal initialization
            self.allow_indels = allow_indels
            if not isinstance(entries, list):
                entries = list(entries)  # Ensure entries is a list
            self._entries = entries  # Store original entries for pickle support
            self._trie = cPrefixTrie(entries, allow_indels)
            self._shared_memory = None
            self._is_shared_owner = False
            # Create Python set for ultra-fast exact matching
            self._exact_set = set(entries)

        # Register cleanup handler once
        if not _cleanup_registered:
            atexit.register(_cleanup_shared_memory)
            _cleanup_registered = True

    def create_shared_memory(self, name: str=None) -> str:
        """
        Create a shared memory block containing this trie's data.
        Returns the name of the shared memory block.

        :param name: Optional name for the shared memory block
        :return: Name of the created shared memory block
        """
        # Serialize the trie data
        data = {
            'entries': self._entries,
            'allow_indels': self.allow_indels
        }
        serialized_data = pickle.dumps(data)

        # Create shared memory block
        try:
            if name:
                shm_block = shm.SharedMemory(name=name, create=True, size=len(serialized_data))
            else:
                shm_block = shm.SharedMemory(create=True, size=len(serialized_data))

            # Copy data to shared memory
            shm_block.buf[:len(serialized_data)] = serialized_data

            # Track for cleanup
            _shared_memory_registry.add(shm_block)

            # Store reference for potential cleanup
            self._shared_memory = shm_block
            self._is_shared_owner = True

            return shm_block.name

        except Exception as e:
            raise RuntimeError(f"Failed to create shared memory: {e}")

    def _load_from_shared_memory(self, name: str):
        """Load trie data from shared memory block"""
        try:
            # Connect to existing shared memory
            shm_block = shm.SharedMemory(name=name, create=False)

            # Deserialize data
            serialized_data = bytes(shm_block.buf)
            data = pickle.loads(serialized_data)

            # Initialize trie
            self.allow_indels = data['allow_indels']
            self._entries = data['entries']
            self._trie = cPrefixTrie(self._entries, self.allow_indels)
            # Create Python set for ultra-fast exact matching
            self._exact_set = set(self._entries)

            # Store reference (but not as owner)
            self._shared_memory = shm_block
            self._is_shared_owner = False

        except Exception as e:
            raise RuntimeError(f"Failed to load from shared memory '{name}': {e}")

    def cleanup_shared_memory(self):
        """Clean up shared memory if this instance owns it"""
        if hasattr(self, '_shared_memory') and self._shared_memory and hasattr(self, '_is_shared_owner') and self._is_shared_owner:
            try:
                self._shared_memory.unlink()
                _shared_memory_registry.discard(self._shared_memory)
            except:
                pass  # Ignore errors
            finally:
                self._shared_memory = None
                self._is_shared_owner = False

    def __getstate__(self):
        """
        Support for pickle serialization.
        Returns the state needed to reconstruct the object.
        """
        return {
            'entries': self._entries,
            'allow_indels': self.allow_indels
        }

    def __setstate__(self, state):
        """
        Support for pickle deserialization.
        Reconstructs the object from the pickled state.
        """
        self.allow_indels = state['allow_indels']
        self._entries = state['entries']
        self._trie = cPrefixTrie(self._entries, self.allow_indels)
        self._shared_memory = None
        self._is_shared_owner = False
        # Create Python set for ultra-fast exact matching
        self._exact_set = set(self._entries)

    def __del__(self):
        """Clean up shared memory on deletion if we own it"""
        try:
            self.cleanup_shared_memory()
        except AttributeError:
            # Object may not be fully initialized
            pass

    def search(self, item: str, correction_budget: int=0) -> tuple[str | None, int]:
        """
        Search for an item in the trie with optional corrections.
        :param item: The string to search for in the trie.
        :param correction_budget: Maximum number of corrections allowed (default is 0).
        :return: A tuple containing the found item and the number of corrections, or (None, -1) if not found.
        """
        # Ultra-fast exact matching using Python set (bypasses all Cython overhead)
        if correction_budget == 0:
            # For exact matching, use pure Python set lookup - fastest possible
            return (item, 0) if item in self._exact_set else (None, -1)

        # For fuzzy matching, first check if it's an exact match in the set
        if item in self._exact_set:
            return (item, 0)

        # Use trie for fuzzy matching only when needed
        found, corrections = self._trie.search(item, correction_budget)
        return found, corrections

    def search_substring(self, target_string: str, correction_budget: int=0) -> tuple[str | None, int, int, int]:
        """
        Search for fuzzy substring matches of trie entries within a target string.

        This method finds any entry from the trie that appears as a fuzzy substring
        within the target string, allowing for insertions, deletions, and substitutions.

        :param target_string: The string to search within for trie entries
        :param correction_budget: Maximum number of edits allowed (default is 0)
        :return: Tuple of (found_string, corrections, start_pos, end_pos) or (None, -1, -1, -1)
                 where start_pos and end_pos indicate the location of the match in target_string

        Example:
            >>> trie = PrefixTrie(["HELLO", "WORLD"], allow_indels=True)
            >>> result = trie.search_substring("AAAAHELLOAAAA", correction_budget=0)
            >>> # Returns: ("HELLO", 0, 4, 9)

            >>> result = trie.search_substring("AAAHELOAAAA", correction_budget=1)
            >>> # Returns: ("HELLO", 1, 3, 8) - found with 1 substitution
        """
        return self._trie.search_substring(target_string, correction_budget)

    def __contains__(self, item: str) -> bool:
        """
        Check if the trie contains the given item.
        :param item: The string to check for presence in the trie.
        :return: True if the item is in the trie, False otherwise.
        """
        # First check in the exact set for ultra-fast exact matching
        if item in self._exact_set:
            return True

        found, _ = self._trie.search(item)
        return found is not None

    def __iter__(self):
        """
        Iterate over the items in the trie.
        :return: An iterator over the items in the trie.
        """
        yield from self._trie.make_iter()

    def __len__(self):
        """
        Get the number of items in the trie.
        :return: The number of items in the trie.
        """
        return self._trie.n_values()

    def __repr__(self):
        """
        String representation of the PrefixTrie.
        :return: A string representation of the PrefixTrie.
        """
        return f"PrefixTrie(n_entries={len(self)}, allow_indels={self.allow_indels})"

    def __str__(self):
        """
        String representation of the PrefixTrie.
        :return: A string representation of the PrefixTrie.
        """
        return f"PrefixTrie with {len(self)} entries, allow_indels={self.allow_indels}"

    def __getitem__(self, item: str) -> str:
        """
        Get the item from the trie.
        :param item: The string to retrieve from the trie.
        :return: The item if found, otherwise raises KeyError.
        """
        found, _ = self._trie.search(item)
        if found is None:
            raise KeyError(f"{item} not found in PrefixTrie")
        return found


# Convenience function for shared memory multiprocessing
def create_shared_trie(entries: list[str], allow_indels: bool=False, name: str=None) -> tuple[PrefixTrie, str]:
    """
    Create a PrefixTrie and put it in shared memory for multiprocessing.

    :param entries: List of strings to add to the trie
    :param allow_indels: Whether to allow insertions/deletions
    :param name: Optional name for shared memory block
    :return: Tuple of (trie_instance, shared_memory_name)
    """
    trie = PrefixTrie(entries, allow_indels)
    shm_name = trie.create_shared_memory(name)
    return trie, shm_name


def load_shared_trie(shared_memory_name: str) -> PrefixTrie:
    """
    Load a PrefixTrie from shared memory.

    :param shared_memory_name: Name of the shared memory block
    :return: PrefixTrie instance loaded from shared memory
    """
    return PrefixTrie([], shared_memory_name=shared_memory_name)


__all__ = ["PrefixTrie", "create_shared_trie", "load_shared_trie"]
