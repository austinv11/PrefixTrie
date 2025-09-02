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

## Performance Characteristics

The implementation is optimized for read-heavy workloads with several key optimizations:

1. **Collapsed terminal nodes** for trivial exact paths
2. **Aggressive caching** of subproblem results during search
3. **Best-case-first search** strategy
4. **Substitution preference** over indels (configurable)
5. **Ultra-fast exact matching** bypassing trie overhead for correction_budget=0

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

## Advanced Search Operations

### Substring Search

Find trie entries that appear as substrings within larger strings:

```python
trie = PrefixTrie(["HELLO", "WORLD"], allow_indels=True)

# Exact substring match
result, corrections, start, end = trie.search_substring("AAAAHELLOAAAA", correction_budget=0)
print(f"Found '{result}' with {corrections} edits at positions {start}:{end}")
# Found 'HELLO' with 0 edits at positions 4:9

# Fuzzy substring match
result, corrections, start, end = trie.search_substring("AAAHELOAAAA", correction_budget=1)
print(f"Found '{result}' with {corrections} edits at positions {start}:{end}")
# Found 'HELLO' with 1 edits at positions 3:8
```

### Longest Prefix Matching

Find the longest prefix from the trie that matches the beginning of a target string:

```python
trie = PrefixTrie(["ACGT", "ACGTA", "ACGTAG"])

# Find longest prefix match
result, start_pos, match_length = trie.longest_prefix_match("ACGTAGGT", min_match_length=4)
print(f"Longest match: '{result}' at position {start_pos}, length {match_length}")
# Longest match: 'ACGTAG' at position 0, length 6

# No match if minimum length not met
result, start_pos, match_length = trie.longest_prefix_match("ACGTTT", min_match_length=7)
print(result)  # None
```

### Counting Fuzzy Matches

Efficiently count the number of unique entries that match a query within a given correction budget, without retrieving the actual strings.

```python
trie = PrefixTrie(["apple", "apply", "apples", "orange"], allow_indels=True)

# Count exact matches
count = trie.search_count("apple", correction_budget=0)
print(f"Found {count} exact match(es) for 'apple'")
# Found 1 exact match(es) for 'apple'

# Count fuzzy matches
# "apple" (0 corrections) + "apply" (1 correction) + "apples" (1 correction)
count = trie.search_count("apple", correction_budget=1)
print(f"Found {count} fuzzy match(es) for 'apple' with budget 1")
# Found 3 fuzzy match(es) for 'apple' with budget 1
```

## Mutable vs Immutable Tries

### Immutable Tries (Default)

Immutable tries are optimized for read-only operations and support shared memory:

```python
# Immutable by default
trie = PrefixTrie(["apple", "banana"], immutable=True)
print(trie.is_immutable())  # True

# Cannot modify immutable tries
try:
    trie.add("cherry")
except RuntimeError as e:
    print(e)  # Cannot modify immutable trie
```

### Mutable Tries

Mutable tries allow dynamic addition and removal of entries (note that mutability incurs performance penalties):

```python
# Create mutable trie
trie = PrefixTrie(["apple"], immutable=False, allow_indels=True)

# Add new entries
success = trie.add("banana")
print(f"Added banana: {success}")  # True
print(f"Trie size: {len(trie)}")   # 2

# Remove entries
success = trie.remove("apple")
print(f"Removed apple: {success}") # True
print(f"Trie size: {len(trie)}")   # 1

# Try to add duplicate
success = trie.add("banana")
print(f"Added duplicate: {success}")  # False

# All search operations work on mutable tries
result, corrections = trie.search("banan", correction_budget=1)
print(result, corrections)  # ("banana", 1)
```

## Multiprocessing Support

PrefixTrie is fully pickle-compatible for easy use with multiprocessing:

```python
import multiprocessing as mp
from prefixtrie import PrefixTrie

def search_worker(trie, query, budget=1):
    """Worker function that uses the trie"""
    return trie.search(query, correction_budget=budget)

# Create trie
entries = [f"barcode_{i:06d}" for i in range(10000)]
trie = PrefixTrie(entries, allow_indels=True)

# Use with multiprocessing (trie is automatically pickled)
if __name__ == "__main__":
    with mp.Pool(processes=4) as pool:
        queries = ["barcode_000123", "barcode_999999", "invalid_code"]
        results = pool.starmap(search_worker, [(trie, q, 2) for q in queries])
        
    for query, (result, corrections) in zip(queries, results):
        print(f"Query: {query} -> Found: {result}, Corrections: {corrections}")
```

## High-Performance Shared Memory

For large tries and intensive multiprocessing workloads, shared memory provides significant performance benefits:

```python
import multiprocessing as mp
from prefixtrie import create_shared_trie, load_shared_trie

def search_worker(shared_memory_name, query, budget=1):
    """Worker that loads trie from shared memory - very fast!"""
    trie = load_shared_trie(shared_memory_name)
    return trie.search(query, correction_budget=budget)

# Create large trie in shared memory
entries = [f"gene_sequence_{i:08d}" for i in range(100000)]
trie, shm_name = create_shared_trie(entries, allow_indels=True)

try:
    if __name__ == "__main__":
        # Multiple processes can efficiently access the same trie
        with mp.Pool(processes=8) as pool:
            queries = ["gene_sequence_00001234", "gene_sequence_99999999"]
            results = pool.starmap(search_worker, [(shm_name, q, 2) for q in queries])
        
        for query, (result, corrections) in zip(queries, results):
            print(f"Query: {query} -> Found: {result}, Corrections: {corrections}")
            
finally:
    # Clean up shared memory
    trie.cleanup_shared_memory()
```

## Standard Dictionary Interface

PrefixTrie supports standard Python container operations:

```python
trie = PrefixTrie(["apple", "banana", "cherry"])

# Length
print(len(trie))  # 3

# Membership testing
print("apple" in trie)   # True
print("grape" in trie)   # False

# Item access
print(trie["banana"])    # "banana"

# Iteration
for item in trie:
    print(item)  # apple, banana, cherry

# String representation
print(repr(trie))  # PrefixTrie(n_entries=3, allow_indels=False)
```

## Installation

### From PyPI (Recommended)

```bash
pip install prefixtrie
```

### Building from Source

Requires a C++ compiler and Cython:

```bash
git clone https://github.com/austinv11/PrefixTrie.git
cd PrefixTrie

# With UV (preferred)
uv sync --group dev
uv pip install -e .

# With pip
pip install -e .
```

#### Building the Documentation
The documentation is built using [MkDocs](https://www.mkdocs.org/).

```bash
# Install documentation dependencies
uv sync --group dev

# Build the site
mkdocs build
```

The generated documentation will be in the `site` directory.

## Development and Testing

```bash
# Install development dependencies
uv sync --group test
uv pip install -e .

# Run tests
pytest test/

# Run benchmarks
python run_benchmark.py
python run_substring_benchmark.py
```

## Performance Notes

1. **Exact matching** (correction_budget=0) uses ultra-fast set lookups
2. **Immutable tries** are faster and more memory-efficient than mutable ones
3. **Shared memory** provides significant speedup for multiprocessing with large tries
4. **Substitutions** are prioritized over insertions/deletions when both are possible
5. The implementation assumes ASCII characters; Unicode support is not guaranteed

## Algorithm Details

- **Fuzzy search** uses dynamic programming with aggressive caching
- **Collapsed nodes** optimize memory usage and search speed
- **Best-case-first** search strategy minimizes unnecessary computation
- **Length bounds** pruning eliminates impossible matches early
- **Alphabet optimization** for immutable tries reduces memory footprint

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/austinv11/PrefixTrie) for issues and pull requests.
