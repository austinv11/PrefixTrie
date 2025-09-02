# Examples

This page contains a collection of examples demonstrating how to use PrefixTrie for various tasks.

## Basic Search Operations

```python
from prefixtrie import PrefixTrie

# Create a trie
trie = PrefixTrie(["ACGT", "ACGG", "ACGC"], allow_indels=True)

# Search with different correction budgets
result, corrections = trie.search("ACGT", correction_budget=0)  # Exact match
print(f"Found '{result}' with {corrections} corrections.")

result, corrections = trie.search("ACGA", correction_budget=1)  # Allow 1 edit
print(f"Found '{result}' with {corrections} corrections.")
```

## Substring Search

Find trie entries that appear as substrings within larger strings.

```python
from prefixtrie import PrefixTrie

trie = PrefixTrie(["HELLO", "WORLD"])

# Find trie entries within larger strings
result, corrections, start, end = trie.search_substring("AAAAHELLOAAAA", correction_budget=0)
print(f"Found '{result}' at positions {start}:{end}")
```

## Longest Prefix Matching

Find the longest prefix from the trie that matches the beginning of a target string.

```python
from prefixtrie import PrefixTrie

trie = PrefixTrie(["ACGT", "ACGTA", "ACGTAG"])

# Find the longest prefix match
result, start_pos, match_length = trie.longest_prefix_match("ACGTAGGT", min_match_length=4)
print(f"Longest match: '{result}' at position {start_pos}, length {match_length}")
```

## Counting Matches

Count how many entries match a query within a given correction budget.

```python
from prefixtrie import PrefixTrie

trie = PrefixTrie(["apple", "apply", "apples", "orange"])

# Count how many entries match a query
count = trie.search_count("apple", correction_budget=1)
print(f"Found {count} match(es) for 'apple' with budget 1.")
```

## Mutable Operations

Create a mutable trie to dynamically add and remove entries.

```python
from prefixtrie import PrefixTrie

# Create a mutable trie for dynamic modifications
trie = PrefixTrie(["apple"], immutable=False, allow_indels=True)
print(f"Initial trie: {list(trie)}")

trie.add("banana")
print(f"After adding 'banana': {list(trie)}")

trie.remove("apple")
print(f"After removing 'apple': {list(trie)}")
```

## Standard Dictionary Interface

PrefixTrie supports standard Python container operations, making it feel like a regular Python dictionary.

```python
from prefixtrie import PrefixTrie

trie = PrefixTrie(["apple", "banana", "cherry"])

# Length
print(f"Size of trie: {len(trie)}")

# Membership testing
print(f"'apple' in trie: {'apple' in trie}")
print(f"'grape' in trie: {'grape' in trie}")

# Item access
print(f"Accessing 'banana': {trie['banana']}")

# Iteration
print("Iterating over trie:")
for item in trie:
    print(f"- {item}")

# String representation
print(f"String representation: {repr(trie)}")
```
