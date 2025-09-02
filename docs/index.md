# Welcome to PrefixTrie

[![PyPI version](https://img.shields.io/pypi/v/prefixtrie.svg)](https://pypi.org/project/prefixtrie/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/austinv11/PrefixTrie/ci.yml?branch=master)](https://github.com/austinv11/PrefixTrie/actions)
[![License](https://img.shields.io/github/license/austinv11/PrefixTrie.svg)](https://github.com/austinv11/PrefixTrie/blob/master/LICENSE)

A high-performance Cython implementation of a prefix trie data structure for efficient fuzzy string matching. Originally designed for RNA barcode matching in bioinformatics applications, but suitable for any use case requiring fast approximate string search.

## Features

- **Ultra-fast exact matching** using optimized Python sets
- **Fuzzy matching** with configurable edit distance (insertions, deletions, substitutions)
- **Substring search** to find trie entries within larger strings
- **Longest prefix matching** for sequence analysis
- **Mutable and immutable** trie variants
- **Multiprocessing support** with pickle compatibility
- **Shared memory** for high-performance parallel processing
- **Memory-efficient** with collapsed node optimization
- **Bioinformatics-optimized** for DNA/RNA/protein sequences

## Basic Usage

```python
from prefixtrie import PrefixTrie

# Create a trie with DNA sequences
trie = PrefixTrie(["ACGT", "ACGG", "ACGC"], allow_indels=True)

# Exact matching
result, corrections = trie.search("ACGT")
print(result, corrections)  # ("ACGT", 0)

# Fuzzy matching with edit distance
result, corrections = trie.search("ACGA", correction_budget=1)
print(result, corrections)  # ("ACGT", 1) - one substitution

result, corrections = trie.search("ACG", correction_budget=1)
print(result, corrections)  # ("ACGT", 1) - one insertion needed

result, corrections = trie.search("ACGTA", correction_budget=1)
print(result, corrections)  # ("ACGT", 1) - one deletion needed

# No match within budget
result, corrections = trie.search("TTTT", correction_budget=1)
print(result, corrections)  # (None, -1)
```

For more advanced usage, please see the [API Reference](./reference.md).