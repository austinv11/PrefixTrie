from ._impl import cPrefixTrie

try:
    from ._version import __version__
except ImportError:
    # Fallback version if _version.py doesn't exist yet
    __version__ = "unknown"


class PrefixTrie:
    """
    Thin wrapper around the cPrefixTrie class to provide a Python interface.
    """

    __slots__ = ("_trie","allow_indels")

    def __init__(self, entries: list[str], allow_indels: bool=False):
        """
        Initialize the PrefixTrie with the given arguments.
        :param entries: List of strings to be added to the trie.
        :param allow_indels: If True, allows insertions and deletions in the trie
        """
        self.allow_indels = allow_indels
        self._trie = cPrefixTrie(entries, allow_indels)

    def search(self, item: str, correction_budget: int=0) -> tuple[str, bool]:
        """
        Search for an item in the trie with optional corrections.
        :param item: The string to search for in the trie.
        :param correction_budget: Maximum number of corrections allowed (default is 0).
        :return: A tuple containing the found item and a boolean indicating if it was an exact match.
        """
        found, exact = self._trie.search(item, correction_budget)
        return found, exact

    def __contains__(self, item: str) -> bool:
        """
        Check if the trie contains the given item.
        :param item: The string to check for presence in the trie.
        :return: True if the item is in the trie, False otherwise.
        """
        found, exact = self._trie.search(item)
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
        found, exact = self._trie.search(item)
        if found is None:
            raise KeyError(f"{item} not found in PrefixTrie")
        return found


__all__ = ["PrefixTrie"]
