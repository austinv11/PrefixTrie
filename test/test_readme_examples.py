#!/usr/bin/env python3
"""
Formal unit tests to verify all README examples work correctly.
This test file validates that all code examples in the README.md work as documented.
"""

import pytest
import tempfile
import os
import multiprocessing as mp
import pyximport

pyximport.install(
    setup_args={"include_dirs": ["../src/prefixtrie"]},
)
from prefixtrie import PrefixTrie, create_shared_trie, load_shared_trie


# Module-level worker functions for multiprocessing tests
# These need to be at module level to be picklable

def search_worker(trie, query, budget=1):
    """Worker function that uses the trie"""
    return trie.search(query, correction_budget=budget)


def shared_memory_worker(shared_memory_name, query, budget=1):
    """Worker that loads trie from shared memory - very fast!"""
    trie = load_shared_trie(shared_memory_name)
    return trie.search(query, correction_budget=budget)


class TestReadmeBasicUsage:
    """Test all basic usage examples from README"""

    def test_basic_dna_sequence_example(self):
        """Test the main DNA sequence example from README"""
        # Create a trie with DNA sequences
        trie = PrefixTrie(["ACGT", "ACGG", "ACGC"], allow_indels=True)

        # Exact matching
        result, corrections = trie.search("ACGT")
        assert result == "ACGT"
        assert corrections == 0

        # Fuzzy matching with edit distance - substitution
        result, corrections = trie.search("ACGA", correction_budget=1)
        assert result == "ACGT"
        assert corrections == 1

        # Fuzzy matching - insertion needed
        result, corrections = trie.search("ACG", correction_budget=1)
        assert result == "ACGT"
        assert corrections == 1

        # Fuzzy matching - deletion needed
        result, corrections = trie.search("ACGTA", correction_budget=1)
        assert result == "ACGT"
        assert corrections == 1

        # No match within budget
        result, corrections = trie.search("TTTT", correction_budget=1)
        assert result is None
        assert corrections == -1


class TestReadmeAdvancedSearchOperations:
    """Test advanced search operations from README"""

    def test_substring_search_examples(self):
        """Test substring search examples from README"""
        trie = PrefixTrie(["HELLO", "WORLD"], allow_indels=True)

        # Exact substring match
        result, corrections, start, end = trie.search_substring("AAAAHELLOAAAA", correction_budget=0)
        assert result == "HELLO"
        assert corrections == 0
        assert start == 4
        assert end == 9

        # Fuzzy substring match
        result, corrections, start, end = trie.search_substring("AAAHELOAAAA", correction_budget=1)
        assert result == "HELLO"
        assert corrections == 1
        assert start == 3
        assert end == 7

    def test_longest_prefix_matching_examples(self):
        """Test longest prefix matching examples from README"""
        trie = PrefixTrie(["ACGT", "ACGTA", "ACGTAG"])

        # Find longest prefix match
        result, start_pos, match_length = trie.longest_prefix_match("ACGTAGGT", min_match_length=4)
        assert result == "ACGTAG"
        assert start_pos == 0
        assert match_length == 6

        # No match if minimum length not met
        result, start_pos, match_length = trie.longest_prefix_match("ACGTTT", min_match_length=7)
        assert result is None
        assert start_pos == -1
        assert match_length == -1


class TestReadmeMutabilityFeatures:
    """Test mutable vs immutable trie examples from README"""

    def test_immutable_tries_default(self):
        """Test immutable tries (default behavior) from README"""
        # Immutable by default
        trie = PrefixTrie(["apple", "banana"], immutable=True)
        assert trie.is_immutable() == True

        # Cannot modify immutable tries
        with pytest.raises(RuntimeError) as exc_info:
            trie.add("cherry")
        assert "Cannot modify immutable trie" in str(exc_info.value)

    def test_mutable_tries_examples(self):
        """Test mutable trie examples from README"""
        # Create mutable trie
        trie = PrefixTrie(["apple"], immutable=False, allow_indels=True)

        # Add new entries
        success = trie.add("banana")
        assert success == True
        assert len(trie) == 2

        # Remove entries
        success = trie.remove("apple")
        assert success == True
        assert len(trie) == 1

        # Try to add duplicate
        success = trie.add("banana")
        assert success == False

        # All search operations work on mutable tries
        result, corrections = trie.search("banan", correction_budget=1)
        assert result == "banana"
        assert corrections == 1


class TestReadmeMultiprocessingSupport:
    """Test multiprocessing examples from README"""

    def test_basic_multiprocessing_example(self):
        """Test the basic multiprocessing example from README"""
        # Create trie (smaller for testing)
        entries = [f"barcode_{i:06d}" for i in range(100)]
        trie = PrefixTrie(entries, allow_indels=True)

        # Use with multiprocessing (trie is automatically pickled)
        with mp.Pool(processes=2) as pool:
            queries = ["barcode_000023", "barcode_000099", "invalid_code"]
            results = pool.starmap(search_worker, [(trie, q, 2) for q in queries])

        # Verify results
        expected_results = [
            ("barcode_000023", 0),
            ("barcode_000099", 0),
            (None, -1)
        ]

        for i, (query, (result, corrections)) in enumerate(zip(queries, results)):
            expected_result, expected_corrections = expected_results[i]
            if expected_result is not None:
                assert result == expected_result
                assert corrections == expected_corrections
            else:
                assert result is None
                assert corrections == -1

    def test_shared_memory_multiprocessing_example(self):
        """Test the shared memory multiprocessing example from README"""
        # Create trie in shared memory (smaller for testing)
        entries = [f"gene_sequence_{i:08d}" for i in range(100)]
        trie, shm_name = create_shared_trie(entries, allow_indels=True)

        try:
            # Multiple processes can efficiently access the same trie
            with mp.Pool(processes=2) as pool:
                queries = ["gene_sequence_00000034", "gene_sequence_00000099"]
                results = pool.starmap(shared_memory_worker, [(shm_name, q, 2) for q in queries])

            # Verify results
            for query, (result, corrections) in zip(queries, results):
                assert result == query  # Should find exact matches
                assert corrections == 0

        finally:
            # Clean up shared memory
            trie.cleanup_shared_memory()


class TestReadmeDictionaryInterface:
    """Test standard dictionary interface examples from README"""

    def test_dictionary_interface_examples(self):
        """Test all dictionary interface examples from README"""
        trie = PrefixTrie(["apple", "banana", "cherry"])

        # Length
        assert len(trie) == 3

        # Membership testing
        assert "apple" in trie
        assert "grape" not in trie

        # Item access
        assert trie["banana"] == "banana"

        # Iteration
        items = list(trie)
        assert set(items) == {"apple", "banana", "cherry"}

        # String representation
        repr_str = repr(trie)
        assert "PrefixTrie" in repr_str
        assert "n_entries=3" in repr_str
        assert "allow_indels=False" in repr_str


class TestReadmeBioinformaticsExamples:
    """Test bioinformatics application examples from README"""

    def test_dna_rna_barcode_matching(self):
        """Test DNA/RNA barcode matching example"""
        # RNA barcodes with fuzzy matching
        barcodes = ["ACGTACGT", "TGCATGCA", "GGCCGGCC", "AATTAATT"]
        trie = PrefixTrie(barcodes, allow_indels=True)

        # Match with sequencing errors
        observed_sequence = "ACGTACGA"  # One base substitution
        result, corrections = trie.search(observed_sequence, correction_budget=2)
        assert result == "ACGTACGT"
        assert corrections == 1

    def test_protein_sequence_analysis(self):
        """Test protein sequence analysis example"""
        # Protein domains
        domains = ["MKLLFY", "ARNDCQ", "EGHILK", "MNPQRS"]
        trie = PrefixTrie(domains, allow_indels=True)

        # Find domain in longer protein sequence
        protein = "XXXMKLLFYYYY"
        result, corrections, start, end = trie.search_substring(protein, correction_budget=1)
        assert result == "MKLLFY"
        assert corrections == 0
        assert start == 3
        assert end == 9

    def test_gene_prefix_analysis(self):
        """Test gene prefix analysis example"""
        # Gene prefixes
        gene_starts = ["ATG", "ATGC", "ATGCA", "ATGCAT"]  # Start codons and extensions
        trie = PrefixTrie(gene_starts)

        # Find longest matching prefix
        sequence = "ATGCATGGG"
        result, pos, length = trie.longest_prefix_match(sequence, min_match_length=3)
        assert result == "ATGCAT"
        assert pos == 0
        assert length == 6


class TestReadmeErrorHandling:
    """Test error handling scenarios mentioned in README"""

    def test_key_error_on_missing_item(self):
        """Test that KeyError is raised for missing items"""
        trie = PrefixTrie(["alpha", "beta"])

        # Accessing existing items should work
        assert trie["alpha"] == "alpha"
        assert trie["beta"] == "beta"

        # Accessing non-existent item should raise KeyError
        with pytest.raises(KeyError) as exc_info:
            _ = trie["gamma"]
        assert "gamma not found in PrefixTrie" in str(exc_info.value)

    def test_shared_memory_immutable_restriction(self):
        """Test that shared memory requires immutable tries"""
        # Mutable trie should not be able to create shared memory
        mutable_trie = PrefixTrie(["hello", "world"], immutable=False)
        with pytest.raises(RuntimeError) as exc_info:
            mutable_trie.create_shared_memory()
        assert "Cannot create shared memory for mutable trie" in str(exc_info.value)

        # Immutable trie should be able to create shared memory
        immutable_trie = PrefixTrie(["hello", "world"], immutable=True)
        try:
            shm_name = immutable_trie.create_shared_memory()
            assert shm_name is not None

            # Should be able to load from shared memory
            loaded_trie = load_shared_trie(shm_name)
            assert len(loaded_trie) == 2
            assert "hello" in loaded_trie
        finally:
            immutable_trie.cleanup_shared_memory()


class TestReadmePerformanceFeatures:
    """Test performance-related features mentioned in README"""

    def test_exact_matching_optimization(self):
        """Test that exact matching (correction_budget=0) is optimized"""
        entries = ["test", "example", "performance"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Exact matching should use fast set lookup
        result, corrections = trie.search("test", correction_budget=0)
        assert result == "test"
        assert corrections == 0

        # Non-exact should return None with correction_budget=0
        result, corrections = trie.search("tset", correction_budget=0)
        assert result is None
        assert corrections == -1

        # Same query should work with correction_budget > 0
        result, corrections = trie.search("tset", correction_budget=2)
        assert result == "test"
        assert corrections == 2

    def test_immutable_vs_mutable_performance_difference(self):
        """Test that immutable and mutable tries both work correctly"""
        entries = ["performance", "test", "comparison"]

        # Immutable trie (default)
        immutable_trie = PrefixTrie(entries, allow_indels=True, immutable=True)
        assert immutable_trie.is_immutable() == True

        # Mutable trie
        mutable_trie = PrefixTrie(entries, allow_indels=True, immutable=False)
        assert mutable_trie.is_immutable() == False

        # Both should give same search results
        test_query = "performnce"  # missing 'a'

        immutable_result = immutable_trie.search(test_query, correction_budget=1)
        mutable_result = mutable_trie.search(test_query, correction_budget=1)

        assert immutable_result == mutable_result
        assert immutable_result[0] == "performance"
        assert immutable_result[1] == 1


class TestReadmeEdgeCases:
    """Test edge cases and special scenarios"""

    def test_empty_trie_behavior(self):
        """Test behavior with empty trie"""
        trie = PrefixTrie([])

        assert len(trie) == 0
        assert "anything" not in trie

        result, corrections = trie.search("test")
        assert result is None
        assert corrections == -1

        # Test iteration on empty trie
        assert list(trie) == []

    def test_single_character_entries(self):
        """Test with single character entries"""
        trie = PrefixTrie(["A", "T", "C", "G"], allow_indels=True)

        # Exact matches
        for char in ["A", "T", "C", "G"]:
            result, corrections = trie.search(char)
            assert result == char
            assert corrections == 0

        # Fuzzy match
        result, corrections = trie.search("X", correction_budget=1)
        assert result in ["A", "T", "C", "G"]
        assert corrections == 1

    def test_very_long_strings(self):
        """Test with very long strings"""
        long_string = "ATCG" * 100  # 400 characters
        entries = [long_string, long_string + "A"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Exact match
        result, corrections = trie.search(long_string)
        assert result == long_string
        assert corrections == 0

        # Fuzzy match
        modified_string = long_string[:-1] + "T"  # Change last character
        result, corrections = trie.search(modified_string, correction_budget=1)
        assert result == long_string
        assert corrections == 1

    def test_special_characters(self):
        """Test with special characters in strings"""
        entries = ["hello!", "test@domain.com", "file_name.txt", "a-b-c"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Exact matches
        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

        # Test membership
        for entry in entries:
            assert entry in trie
            assert trie[entry] == entry


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    print("Running README example verification tests...")

    # Run a few key tests to verify everything works
    test_basic = TestReadmeBasicUsage()
    test_basic.test_basic_dna_sequence_example()
    print("✓ Basic usage examples work")

    test_advanced = TestReadmeAdvancedSearchOperations()
    test_advanced.test_substring_search_examples()
    test_advanced.test_longest_prefix_matching_examples()
    print("✓ Advanced search operations work")

    test_dict = TestReadmeDictionaryInterface()
    test_dict.test_dictionary_interface_examples()
    print("✓ Dictionary interface works")

    test_mutability = TestReadmeMutabilityFeatures()
    test_mutability.test_immutable_tries_default()
    test_mutability.test_mutable_tries_examples()
    print("✓ Mutability features work")

    print("\n🎉 All README examples verified successfully!")
    print("Run 'pytest test/test_readme_examples.py' for the full test suite.")
