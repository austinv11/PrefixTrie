# API Reference

This page contains the complete API documentation for PrefixTrie, automatically generated from the source code docstrings.

::: prefixtrie.PrefixTrie
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      group_by_category: true
      show_bases: false
      show_inheritance_diagram: false
      heading_level: 2

---

# Usage Examples

The following examples demonstrate how to use the PrefixTrie API with your specific argument syntax.

## Basic Search Operations

```python
from prefixtrie import PrefixTrie

# Create a trie
trie = PrefixTrie(["ACGT", "ACGG", "ACGC"], allow_indels=True)

# Search with different correction budgets
result, corrections = trie.search("ACGT", correction_budget=0)  # Exact match
result, corrections = trie.search("ACGA", correction_budget=1)  # Allow 1 edit
```

## Substring Search

```python
# Find trie entries within larger strings
result, corrections, start, end = trie.search_substring("AAAAHELLOAAAA", correction_budget=0)
```

## Longest Prefix Matching

```python
# Find the longest prefix match
result, start_pos, match_length = trie.longest_prefix_match("ACGTAGGT", min_match_length=4)
```

## Counting Matches

```python
# Count how many entries match a query
count = trie.search_count("apple", correction_budget=1)
```

## Mutable Operations

```python
# Create a mutable trie for dynamic modifications
trie = PrefixTrie(["apple"], immutable=False, allow_indels=True)
trie.add("banana")
trie.remove("apple")
```
