# Getting Started

This guide will walk you through the installation and basic usage of PrefixTrie.

## Installation

You can install PrefixTrie directly from PyPI using pip:

```bash
pip install prefixtrie
```

This will install the latest stable version of the library.

## Your First Trie

Let's create your first prefix trie and perform a simple search.

```python
from prefixtrie import PrefixTrie

# 1. Create a trie with a list of strings
trie = PrefixTrie(["ACGT", "ACGG", "ACGC"])

# 2. Perform an exact search
# The search function returns the matched string and the number of corrections.
result, corrections = trie.search("ACGT")

print(f"Found '{result}' with {corrections} corrections.")
# Expected output: Found 'ACGT' with 0 corrections.
```

## Fuzzy Matching

PrefixTrie shines when it comes to fuzzy matching. You can specify a `correction_budget` to allow for a certain number of edits (insertions, deletions, or substitutions).

By default, `allow_indels` is `False`, so only substitutions are allowed. To allow insertions and deletions, you need to set `allow_indels=True` when creating the trie.

```python
from prefixtrie import PrefixTrie

# Create a trie that allows insertions and deletions
trie = PrefixTrie(["ACGT", "ACGG", "ACGC"], allow_indels=True)

# 1. Fuzzy matching with one substitution
result, corrections = trie.search("ACGA", correction_budget=1)
print(f"Found '{result}' with {corrections} correction(s).")
# Expected output: Found 'ACGT' with 1 correction(s).

# 2. Fuzzy matching with one insertion
result, corrections = trie.search("ACG", correction_budget=1)
print(f"Found '{result}' with {corrections} correction(s).")
# Expected output: Found 'ACGT' with 1 correction(s).

# 3. Fuzzy matching with one deletion
result, corrections = trie.search("ACGTA", correction_budget=1)
print(f"Found '{result}' with {corrections} correction(s).")
# Expected output: Found 'ACGT' with 1 correction(s).
```

## Additional Examples
This page has gone over the absolute basics, to see more examples and use cases, check out the [Examples](examples.md) page.