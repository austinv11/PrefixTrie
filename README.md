# PrefixTrie

[![PyPI version](https://img.shields.io/pypi/v/PrefixTrie.svg)](https://pypi.org/project/PrefixTrie/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/austinv11/PrefixTrie/ci.yml?branch=master)](https://github.com/austinv11/PrefixTrie/actions)
[![License](https://img.shields.io/github/license/austinv11/PrefixTrie.svg)](https://github.com/austinv11/PrefixTrie/blob/master/LICENSE)

This is a straightforward, read-only, implementation of a Prefix Trie to perform efficient fuzzy string matches.


Note that this is intentionally kept simple and does not include more advanced optimizations (like considering semantic character differences).
Originally, this was meant to only deal with RNA barcode matching. As a result, keep in mind the following:

1. The implementation does not attempt to support non-ASCII characters. It may work in some cases, but I won't make any behavioral guarantees.
2. We assume that insertion/deletions are rare compared to substitutions, so if you enable indel support, you may get suboptimal results when there are multiple possible matches.


## Implementation details in short
We optimize for read-only use cases, so when Tries are initialized, they do some preprocessing to make searches faster.
This comes at the cost of slightly higher memory usage and longer initialization times. Since this is meant to be read-only,
we don't implement methods to re-optimize the trie. Feel free to make a PR if you need that functionality, but I don't intend to add mutability.
The main optimizations are:
1. Each node recalls collapsed terminal nodes if there is a trivial exact path.
2. The search aggressively caches results of subproblems to avoid redundant searches.
3. Best case search is performed first, so we assume that most searches should not require complex processing.
4. We assume that insertions/deletions are slightly less likely than substitutions so we prioritize substitutions over indels when both are enabled.

## Basic Usage

```python
from prefixtrie import PrefixTrie
trie = PrefixTrie(["ACGT", "ACGG", "ACGC"], allow_indels=True)
print(trie.search("ACGT"))
>> ("ACGT", True)  # Exact match
print(trie.search("ACGA", max_substitutions=1))
>> ("ACGT", False)  # One substitution away
print(trie.search("ACG", max_substitutions=1))
>> ("ACGT", False)  # One insertion away
print(trie.search("ACGTA", max_substitutions=1))
>> ("ACGT", False)  # One deletion away
print(trie.search("AG", max_substitutions=1))
>> None  # No match
```

## Installation

Pip (Recommended):
```bash
pip install PrefixTrie
```

From Source (ensure you have a C++ compiler and Cython installed):
```bash
git clone https://github.com/austinv11/PrefixTrie.git
cd PrefixTrie
# With UV (preferred)
uv sync --group dev
uv pip install -e .
# Without UV
pip install -e .
```

## Testing
To run the tests, ensure you have `pytest` installed and run:
```bash
uv sync --group test
uv pip install -e .
pytest tests/
```


