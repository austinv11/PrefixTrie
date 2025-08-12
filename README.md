# PrefixTrie

[![PyPI version](https://img.shields.io/pypi/v/PrefixTrie.svg)](https://pypi.org/project/PrefixTrie/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/austinv11/PrefixTrie/ci.yml?branch=master)](https://github.com/austinv11/PrefixTrie/actions)
[![License](https://img.shields.io/github/license/austinv11/PrefixTrie.svg)](https://github.com/austinv11/PrefixTrie/blob/master/LICENSE)

This is a straightforward, read-only, implementation of a Prefix Trie to perform efficient fuzzy string matches.


Note that this is intentionally kept simple and does not include more advanced optimizations (like considering semantic character differences).
Originally, this was meant to only deal with RNA barcode matching. As a result, keep in mind the following:

1. The implementation does not attempt to support non-ASCII characters. It may work in some cases, but I won't make any behavioral guarantees.
2. We assume that insertion/deletions are slightly more rare compared to substitutions, so if you enable indel support, you may get suboptimal results when there are multiple possible matches.


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
print(trie.search("ACGA", correction_budget=1))
>> ("ACGT", False)  # One substitution away
print(trie.search("ACG", correction_budget=1))
>> ("ACGT", False)  # One insertion away
print(trie.search("ACGTA", correction_budget=1))
>> ("ACGT", False)  # One deletion away
print(trie.search("AG", correction_budget=1))
>> (None, False)  # No match
```

## Multiprocessing Support

PrefixTrie is also pickle-compatible for easy use with multiprocessing:

```python
import multiprocessing as mp
from prefixtrie import PrefixTrie

def search_worker(trie, query):
    """Worker function that uses the trie"""
    return trie.search(query, correction_budget=1)

# Create trie
entries = [f"barcode_{i:06d}" for i in range(10000)]
trie = PrefixTrie(entries, allow_indels=True)

# Use with multiprocessing (trie is automatically pickled)
with mp.Pool(processes=4) as pool:
    queries = ["barcode_000123", "barcode_999999", "invalid_code"]
    results = pool.starmap(search_worker, [(trie, q) for q in queries])
    
for query, (result, exact) in zip(queries, results):
    print(f"Query: {query} -> Found: {result}, Exact: {exact}")
```

## High-Performance Shared Memory

For large tries and intensive multiprocessing, use shared memory for better performance:

```python
import multiprocessing as mp
from prefixtrie import create_shared_trie, load_shared_trie

def search_worker(shared_memory_name, query):
    """Worker function that loads trie from shared memory"""
    trie = load_shared_trie(shared_memory_name)  # Fast loading!
    return trie.search(query, correction_budget=1)

# Create large trie in shared memory
entries = [f"gene_sequence_{i:08d}" for i in range(100000)]
trie, shm_name = create_shared_trie(entries, allow_indels=True)

try:
    # Use with multiprocessing - much faster for large tries!
    with mp.Pool(processes=8) as pool:
        queries = ["gene_sequence_00001234", "gene_sequence_99999999", "mutated_sequence"]
        results = pool.starmap(search_worker, [(shm_name, q) for q in queries])
    
    for query, (result, exact) in zip(queries, results):
        print(f"Query: {query} -> Found: {result}, Exact: {exact}")
        
finally:
    # Clean up shared memory
    trie.cleanup_shared_memory()
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
