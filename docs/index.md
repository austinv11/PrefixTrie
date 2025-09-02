# PrefixTrie

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

## Quick Start

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

## Installation

```bash
pip install prefixtrie
```

## Documentation

For detailed API documentation, see the [API Reference](reference.md).

## Performance

PrefixTrie is highly optimized and typically outperforms similar fuzzy matching libraries:

- **Search Performance**: Substantially faster than RapidFuzz, TheFuzz, and SymSpell
- **Substring Search**: At least on par with fuzzysearch and regex, often faster
- **Memory Efficiency**: Collapsed node optimization reduces memory footprint
- **Parallel Processing**: Full pickle support for multiprocessing workflows

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/austinv11/PrefixTrie/blob/master/LICENSE) file for details.
