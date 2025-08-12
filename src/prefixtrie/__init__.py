from ._impl import cPrefixTrie as PrefixTrie

try:
    from ._version import __version__
except ImportError:
    # Fallback version if _version.py doesn't exist yet
    __version__ = "unknown"

__all__ = ["PrefixTrie"]
