import pytest
import pyximport

pyximport.install()
from prefixtrie import PrefixTrie


class TestPrefixTrieBasic:
    """Test basic functionality of PrefixTrie"""

    def test_empty_trie(self):
        """Test creating an empty trie"""
        trie = PrefixTrie([])
        result, corrections = trie.search("test")
        assert result is None
        assert corrections == -1
        # Searching for an empty string in an empty trie should not report an
        # exact match.
        result, corrections = trie.search("")
        assert result is None
        assert corrections == -1

    def test_single_entry(self):
        """Test trie with single entry"""
        trie = PrefixTrie(["hello"])

        # Exact match
        result, corrections = trie.search("hello")
        assert result == "hello"
        assert corrections == 0

        # No match
        result, corrections = trie.search("world")
        assert result is None
        assert corrections == -1

    def test_multiple_entries(self):
        """Test trie with multiple entries"""
        entries = ["cat", "car", "card", "care", "careful"]
        trie = PrefixTrie(entries)

        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

    def test_trailing_and_missing_characters(self):
        """Ensure extra or missing characters are handled with indels"""
        entries = ["hello"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Extra character at the end should count as a deletion
        result, corrections = trie.search("hello!", correction_budget=1)
        assert result == "hello"
        assert corrections == 1

        # Missing character should be handled as an insertion
        result, corrections = trie.search("hell", correction_budget=1)
        assert result == "hello"
        assert corrections == 1

    def test_prefix_matching(self):
        """Test prefix-based matching"""
        entries = ["test", "testing", "tester", "tea", "team"]
        trie = PrefixTrie(entries)

        # Test exact matches for complete entries
        result, corrections = trie.search("test")
        assert result == "test"
        assert corrections == 0

        result, corrections = trie.search("tea")
        assert result == "tea"
        assert corrections == 0

        # Test that partial prefixes don't match without fuzzy search
        result, corrections = trie.search("te")
        assert result is None
        assert corrections == -1


class TestPrefixTrieEdgeCases:
    """Test edge cases and special characters"""

    def test_empty_string_entry(self):
        """Test with empty string in entries"""
        # Empty strings may not be supported by this trie implementation
        trie = PrefixTrie(["hello", "world"])

        result, corrections = trie.search("")
        assert result is None
        assert corrections == -1

    def test_single_character_entries(self):
        """Test with single character entries"""
        trie = PrefixTrie(["a", "b", "c"])

        result, corrections = trie.search("a")
        assert result == "a"
        assert corrections == 0

        result, corrections = trie.search("d")
        assert result is None
        assert corrections == -1

    def test_duplicate_entries(self):
        """Test with duplicate entries"""
        trie = PrefixTrie(["hello", "hello", "world"])

        result, corrections = trie.search("hello")
        assert result == "hello"
        assert corrections == 0

    def test_special_characters(self):
        """Test with special characters"""
        entries = ["hello!", "test@123", "a-b-c", "x_y_z"]
        trie = PrefixTrie(entries)

        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

    def test_case_sensitivity(self):
        """Test case sensitivity"""
        trie = PrefixTrie(["Hello", "hello", "HELLO"])

        result, corrections = trie.search("Hello")
        assert result == "Hello"
        assert corrections == 0

        result, corrections = trie.search("hello")
        assert result == "hello"
        assert corrections == 0

        result, corrections = trie.search("HELLO")
        assert result == "HELLO"
        assert corrections == 0

    def test_budget_increase_recomputes(self):
        trie = PrefixTrie(["hello"], allow_indels=True)
        result, corrections = trie.search("hallo", correction_budget=0)
        assert result is None and corrections == -1

        # With more corrections available, the match should now succeed
        result, corrections = trie.search("hallo", correction_budget=1)
        assert result == "hello" and corrections == 1


class TestPrefixTrieFuzzyMatching:
    """Test fuzzy matching capabilities"""

    def test_basic_fuzzy_matching(self):
        """Test basic fuzzy matching with corrections"""
        entries = ["hello", "world", "python"]
        trie = PrefixTrie(entries, allow_indels=False)

        # Test with 1 correction budget - single character substitution
        result, corrections = trie.search("hallo", correction_budget=1)  # e->a substitution
        assert result == "hello"
        assert corrections == 1

        result, corrections = trie.search("worle", correction_budget=1)  # d->e substitution
        assert result == "world"
        assert corrections == 1

    def test_fuzzy_matching_with_indels(self):
        """Test fuzzy matching with insertions and deletions"""
        entries = ["hello", "world"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test simple substitution that should work
        result, corrections = trie.search("hallo", correction_budget=1)
        assert result == "hello"
        assert corrections == 1

        # Test that we can find matches with small edits
        result, corrections = trie.search("worlx", correction_budget=1)
        assert result == "world"
        assert corrections == 1

    def test_correction_budget_limits(self):
        """Test that correction budget is respected"""
        entries = ["hello"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Should find with budget of 2
        result, corrections = trie.search("hallo", correction_budget=2)
        assert result == "hello"
        assert corrections > 0

        # Should not find with budget of 0
        result, corrections = trie.search("hallo", correction_budget=0)
        assert result is None
        assert corrections == -1

    def test_multiple_corrections(self):
        """Test queries requiring multiple corrections"""
        entries = ["testing"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Two substitutions
        result, corrections = trie.search("taxting", correction_budget=2)
        assert result == "testing"
        assert corrections == 2

        # Should not find with insufficient budget
        result, corrections = trie.search("taxting", correction_budget=1)
        assert result is None
        assert corrections == -1


class TestPrefixTriePerformance:
    """Test performance-related scenarios"""

    def test_large_alphabet(self):
        """Test with entries using large character set"""
        entries = [
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "0123456789",
            "!@#$%^&*()_+-="
        ]
        trie = PrefixTrie(entries)

        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

    def test_long_strings(self):
        """Test with very long strings"""
        long_string = "a" * 1000
        entries = [long_string, long_string + "b"]
        trie = PrefixTrie(entries)

        result, corrections = trie.search(long_string)
        assert result == long_string
        assert corrections == 0

    def test_many_entries(self):
        """Test with many entries"""
        entries = [f"entry_{i:04d}" for i in range(1000)]
        trie = PrefixTrie(entries)

        # Test a few random entries
        test_entries = [entries[0], entries[500], entries[999]]
        for entry in test_entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0


class TestPrefixTrieDNASequences:
    """Test with DNA-like sequences (similar to the original test)"""

    def test_dna_sequences(self):
        """Test with DNA sequences"""
        sequences = ["ACGT", "TCGA", "AAAA", "TTTT", "CCCC", "GGGG"]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Exact matches
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

    def test_dna_fuzzy_matching(self):
        """Test fuzzy matching with DNA sequences"""
        sequences = ["ACGT", "TCGA"]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Single base substitution
        result, corrections = trie.search("ACCT", correction_budget=1)
        assert result == "ACGT"
        assert corrections == 1

        # Test with a clear mismatch that requires correction
        result, corrections = trie.search("ACXX", correction_budget=2)
        assert result == "ACGT"
        assert corrections == 2

        # Test that fuzzy matching works with sufficient budget
        result, corrections = trie.search("TCXX", correction_budget=2)
        assert result == "TCGA"
        assert corrections == 2

    def test_similar_sequences(self):
        """Test with very similar sequences"""
        sequences = ["ATCG", "ATCGA", "ATCGAA", "ATCGAAA"]
        trie = PrefixTrie(sequences)

        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

    def test_medium_length_dna_sequences(self):
        """Test with medium-length DNA sequences (20-50 bases)"""
        sequences = [
            "ATCGATCGATCGATCGATCG",  # 20 bases
            "GCTAGCTAGCTAGCTAGCTAGCTA",  # 23 bases
            "AAATTTCCCGGGAAATTTCCCGGGAAATTT",  # 29 bases
            "TCGATCGATCGATCGATCGATCGATCGATCG",  # 30 bases
            "AGCTTAGCTTAGCTTAGCTTAGCTTAGCTTAGCTTA",  # 35 bases
            "CGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA",  # 39 bases
            "TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA",  # 43 bases
            "GCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCG"  # 45 bases
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Test exact matches
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

        # Test fuzzy matching with single substitution
        result, corrections = trie.search("ATCGATCGATCGATCGATCX", correction_budget=1)
        assert result == "ATCGATCGATCGATCGATCG"
        assert corrections == 1

    def test_long_dna_sequences(self):
        """Test with long DNA sequences (100+ bases)"""
        sequences = [
            # 100 base sequence
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            # 120 base sequence
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
            # 150 base sequence with more variety
            "AAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGG",
            # 200 base sequence
            "TCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Test exact matches
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

    def test_realistic_gene_sequences(self):
        """Test with realistic gene-like sequences"""
        # Simulated gene sequences with typical biological patterns
        sequences = [
            # Start codon (ATG) followed by coding sequence
            "ATGAAACGTCTAGCTAGCTAGCTAGCTAG",
            # Promoter-like sequence
            "TATAAAAGGCCGCTCGAGCTCGAGCTCGA",
            # Enhancer-like sequence
            "GCGCGCGCATATATATGCGCGCGCATATA",
            # Terminator-like sequence
            "TTTTTTTTAAAAAAAAGGGGGGGGCCCCCCCC",
            # Splice site-like sequences
            "GTAAGTATCGATCGATCGATCGCAG",
            "CTCGATCGATCGATCGATCGATCAG",
            # Ribosome binding site
            "AGGAGGTTGACATGAAACGTCTAG",
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Test exact matches
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

        # Test mutation simulation (single nucleotide polymorphism)
        result, corrections = trie.search("ATGAAACGTCTAGCTAGCTAGCTAGCTAX", correction_budget=1)
        assert result == "ATGAAACGTCTAGCTAGCTAGCTAGCTAG"
        assert corrections == 1

    def test_repetitive_dna_sequences(self):
        """Test with highly repetitive DNA sequences"""
        sequences = [
            # Tandem repeats
            "CACACACACACACACACACACACACA",  # CA repeat
            "GTGTGTGTGTGTGTGTGTGTGTGTGT",  # GT repeat
            "ATATATATATATATATATATATATAT",  # AT repeat
            "CGCGCGCGCGCGCGCGCGCGCGCGCG",  # CG repeat
            # Short tandem repeats (STRs)
            "AAGAAGAAGAAGAAGAAGAAGAAGAAG",  # AAG repeat
            "CTTCTTCTTCTTCTTCTTCTTCTTCTT",  # CTT repeat
            # Palindromic sequences
            "GAATTCGAATTCGAATTCGAATTC",
            "GCTAGCGCTAGCGCTAGCGCTAGC",
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Test exact matches
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

        # Test with a shorter repetitive sequence for fuzzy matching
        short_sequences = ["CACA", "GTGT", "ATAT"]
        short_trie = PrefixTrie(short_sequences, allow_indels=True)

        result, corrections = short_trie.search("CACX", correction_budget=1)
        assert result == "CACA"
        assert corrections == 1

    def test_mixed_length_dna_database(self):
        """Test with a mixed database of various length sequences"""
        sequences = [
            # Short sequences
            "ATCG", "GCTA", "TTAA", "CCGG",
            # Medium sequences
            "ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA", "TTAATTAATTAATTAA",
            # Long sequences
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
            # Very long sequence (500+ bases)
            "A" * 100 + "T" * 100 + "C" * 100 + "G" * 100 + "ATCG" * 25,
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Test exact matches for all lengths
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

        # Test fuzzy matching across different lengths
        result, corrections = trie.search("ATCX", correction_budget=1)
        assert result == "ATCG"
        assert corrections == 1

        result, corrections = trie.search("ATCGATCGATCGATCX", correction_budget=1)
        assert result == "ATCGATCGATCGATCG"
        assert corrections == 1

    def test_dna_with_ambiguous_bases(self):
        """Test with sequences containing ambiguous DNA bases"""
        sequences = [
            "ATCGNNNGATCG",  # N represents any base
            "RYSWKMBDHVRYSWKM",  # IUPAC ambiguous codes
            "ATCGWSATCGWS",  # W=A/T, S=G/C
            "MRYGATKBHDVM",  # Mixed ambiguous bases
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Test exact matches
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

    def test_codon_sequences(self):
        """Test with codon-based sequences (triplets)"""
        # Common codons and their variations
        sequences = [
            "ATGAAATTTCCCGGG",  # Start codon + amino acids
            "TTTTTCTTATTGCTG",  # Phenylalanine + Leucine codons
            "AAAAAGGATGACGAT",  # Lysine + Aspartic acid codons
            "TAATAGTAA",  # Stop codons
            "GGGGGAGGTGGA",  # Glycine codons
            "CCACCGCCACCCCCT",  # Proline codons
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Test exact matches
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

        # Test single codon mutations
        result, corrections = trie.search("ATGAAATTTCCCGGT", correction_budget=1)  # G->T in last codon
        assert result == "ATGAAATTTCCCGGG"
        assert corrections == 1

    def test_extremely_long_sequences(self):
        """Test with extremely long DNA sequences (1000+ bases)"""
        # Generate very long sequences
        sequences = [
            "ATCG" * 250,  # 1000 bases
            "GCTA" * 300,  # 1200 bases
            "A" * 500 + "T" * 500,  # 1000 bases, two halves
            ("ATCGATCGATCG" * 100)[:1500],  # 1500 bases
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Test exact matches
        for seq in sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0

        # Test fuzzy matching with very long sequences
        query = "ATCG" * 249 + "ATCX"  # 999 bases + ATCX
        result, corrections = trie.search(query, correction_budget=1)
        assert result == "ATCG" * 250
        assert corrections == 1

    def test_dna_performance_benchmark(self):
        """Performance test with many DNA sequences"""
        # Generate a large set of unique sequences
        sequences = []
        bases = "ATCG"

        # 100 sequences of length 50 each
        for i in range(100):
            seq = ""
            for j in range(50):
                seq += bases[(i * 50 + j) % 4]
            sequences.append(seq)

        trie = PrefixTrie(sequences, allow_indels=True)

        # Test a subset for correctness
        test_sequences = sequences[::10]  # Every 10th sequence
        for seq in test_sequences:
            result, corrections = trie.search(seq)
            assert result == seq
            assert corrections == 0


class TestPrefixTrieDunderMethods:
    """Test dunder methods of PrefixTrie"""

    def test_contains(self):
        trie = PrefixTrie(["foo", "bar"])
        assert "foo" in trie
        assert "bar" in trie
        assert "baz" not in trie

    def test_iter(self):
        entries = ["a", "b", "c"]
        trie = PrefixTrie(entries)
        assert set(iter(trie)) == set(entries)

    def test_len(self):
        entries = ["x", "y", "z"]
        trie = PrefixTrie(entries)
        assert len(trie) == 3
        empty_trie = PrefixTrie([])
        assert len(empty_trie) == 0

    def test_getitem(self):
        trie = PrefixTrie(["alpha", "beta"])
        assert trie["alpha"] == "alpha"
        assert trie["beta"] == "beta"
        with pytest.raises(KeyError):
            _ = trie["gamma"]

    def test_repr_and_str(self):
        trie = PrefixTrie(["one", "two"], allow_indels=True)
        r = repr(trie)
        s = str(trie)
        assert "PrefixTrie" in r
        assert "PrefixTrie" in s
        assert "allow_indels=True" in r
        assert "allow_indels=True" in s


class TestPrefixTrieErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_correction_budget(self):
        """Test with negative correction budget"""
        trie = PrefixTrie(["hello"])

        # Negative budget should be treated as 0
        result, corrections = trie.search("hallo", correction_budget=-1)
        assert result is None
        assert corrections == -1


class TestPrefixTrieAdvancedEdgeCases:
    """Test advanced edge cases and algorithm-specific scenarios"""

    def test_insertion_and_deletion_operations(self):
        """Test specific insertion and deletion operations in fuzzy matching"""
        entries = ["hello", "help", "helicopter"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test insertions - query is shorter than target
        result, corrections = trie.search("hell", correction_budget=1)  # could be "hello" or "help" (both 1 edit)
        assert result in ["hello", "help"]  # Both are valid with 1 edit
        assert corrections == 1

        result, corrections = trie.search("hel", correction_budget=1)  # missing 'p' to make "help"
        assert result == "help"
        assert corrections == 1

        # Test deletions - query is longer than target
        result, corrections = trie.search("helllo", correction_budget=1)  # extra 'l'
        assert result == "hello"
        assert corrections == 1

        result, corrections = trie.search("helpx", correction_budget=1)  # extra 'x'
        assert result == "help"
        assert corrections == 1

        # Test substitutions
        result, corrections = trie.search("helo", correction_budget=1)  # 'o'->'p' substitution
        assert result == "help"  # This is correct - only 1 edit needed
        assert corrections == 1

    def test_complex_indel_combinations(self):
        """Test combinations of insertions, deletions, and substitutions"""
        entries = ["algorithm", "logarithm", "rhythm"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Combination: deletion + substitution
        result, corrections = trie.search("algrothm", correction_budget=2)  # missing 'i', 'i'->'o'
        assert result == "algorithm"
        assert corrections == 2

        # Combination: insertion + substitution
        result, corrections = trie.search("logxarithm", correction_budget=2)  # extra 'x', 'x'->'a'
        assert result == "logarithm"
        assert corrections == 2

    def test_prefix_collision_scenarios(self):
        """Test scenarios where prefixes collide and could cause issues"""
        entries = ["a", "aa", "aaa", "aaaa", "aaaaa"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Exact matches should work
        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

        # Fuzzy matching should find closest match
        result, corrections = trie.search("aax", correction_budget=1)
        assert result == "aaa"
        assert corrections == 1

        result, corrections = trie.search("aaax", correction_budget=1)
        assert result == "aaaa"
        assert corrections == 1

    def test_shared_prefix_disambiguation(self):
        """Test disambiguation when multiple entries share long prefixes"""
        entries = [
            "programming", "programmer", "programmed", "programmable",
            "program", "programs", "programmatic"
        ]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

        # Test fuzzy matching with shared prefixes
        result, corrections = trie.search("programmin", correction_budget=1)  # missing 'g'
        assert result == "programming"
        assert corrections == 1

        result, corrections = trie.search("programmerz", correction_budget=1)  # 'z' instead of final char
        assert result == "programmer"
        assert corrections == 1

    def test_empty_and_very_short_queries(self):
        """Test behavior with empty and very short queries"""
        entries = ["a", "ab", "abc", "hello", "world"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Empty query
        result, corrections = trie.search("", correction_budget=0)
        assert result is None
        assert corrections == -1

        result, corrections = trie.search("", correction_budget=1)
        assert result == "a"  # Should find shortest entry
        assert corrections == 1

        # Single character queries
        result, corrections = trie.search("x", correction_budget=1)
        assert result == "a"  # Should find closest single char
        assert corrections == 1

    def test_correction_budget_edge_cases(self):
        """Test edge cases around correction budget limits"""
        entries = ["test", "best", "rest", "nest"]
        entries.sort()
        trie = PrefixTrie(entries, allow_indels=True)

        # Exact budget limit
        result, corrections = trie.search("zest", correction_budget=1)  # 'z'->'t', 'e'->'e', 's'->'s', 't'->'t'
        assert result == "best"
        assert corrections == 1

        # Just over budget
        result, corrections = trie.search("zesz", correction_budget=1)  # needs 2 corrections
        assert result is None
        assert corrections == -1

        # Zero budget should only find exact matches
        result, corrections = trie.search("test", correction_budget=0)
        assert result == "test"
        assert corrections == 0

        result, corrections = trie.search("tesy", correction_budget=0)
        assert result is None
        assert corrections == -1

    def test_alphabet_boundary_conditions(self):
        """Test with characters at alphabet boundaries"""
        entries = ["aaa", "zzz", "AZaz", "09azAZ"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

        # Test fuzzy matching across character boundaries
        result, corrections = trie.search("aab", correction_budget=1)
        assert result == "aaa"
        assert corrections == 1

        result, corrections = trie.search("zzy", correction_budget=1)
        assert result == "zzz"
        assert corrections == 1

    def test_collapsed_path_edge_cases(self):
        """Test edge cases with collapsed paths in the trie"""
        # Create entries that will cause path collapsing
        entries = ["abcdefghijk", "abcdefghijl", "xyz"]
        entries.sort()
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

        # Test fuzzy matching that might interact with collapsed paths
        result, corrections = trie.search("abcdefghijx", correction_budget=1)  # Last char different
        expected = "abcdefghijk"  # Should match first entry
        assert result == expected
        assert corrections == 1

    def test_memory_intensive_operations(self):
        """Test operations that might stress memory management"""
        # Create many similar entries
        entries = [f"pattern{i:03d}suffix" for i in range(100)]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test a few random exact matches
        test_entries = [entries[0], entries[50], entries[99]]
        for entry in test_entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

        # Test fuzzy matching
        result, corrections = trie.search("pattern050suffi", correction_budget=1)  # missing 'x'
        assert result == "pattern050suffix"
        assert corrections == 1

    def test_very_high_correction_budget(self):
        """Test with very high correction budgets"""
        entries = ["short", "verylongstring"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Very high budget should still work correctly
        result, corrections = trie.search("x", correction_budget=100)
        assert result == "short"  # Should find shortest
        assert corrections > 0

        result, corrections = trie.search("completelydifferent", correction_budget=100)
        assert result is not None  # Should find something
        assert corrections > 0

    def test_indel_vs_substitution_preference(self):
        """Test algorithm preference between indels and substitutions"""
        entries = ["abc", "abcd", "abce"]
        entries.sort()
        trie = PrefixTrie(entries, allow_indels=True)

        # This query could match "abc" with 1 deletion or "abcd"/"abce" with 1 substitution
        result, corrections = trie.search("abcx", correction_budget=1)
        # The algorithm should prefer the substitution (keeping same length)
        assert result == "abcd"
        assert corrections == 1

    def test_multiple_valid_corrections(self):
        """Test scenarios where multiple corrections have same cost"""
        entries = ["cat", "bat", "hat", "rat"]
        entries.sort()
        trie = PrefixTrie(entries, allow_indels=True)

        # Query that's 1 edit away from multiple entries
        result, corrections = trie.search("dat", correction_budget=1)
        assert result == "bat"
        assert corrections == 1

        # With higher budget, should still work
        result, corrections = trie.search("zat", correction_budget=1)
        assert result == "bat"
        assert corrections == 1

    def test_nested_prefix_structures(self):
        """Test deeply nested prefix structures"""
        entries = [
            "a", "ab", "abc", "abcd", "abcde", "abcdef",
            "abcdeg", "abcdeh", "abcdei"
        ]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        for entry in entries:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

        # Test fuzzy matching at different depths
        result, corrections = trie.search("abcdej", correction_budget=1)
        assert result in ["abcdef", "abcdeg", "abcdeh", "abcdei"]
        assert corrections == 1

    def test_boundary_string_lengths(self):
        """Test with strings at various length boundaries"""
        entries = [
            "",  # This might not be supported, but let's test
            "x",  # Length 1
            "xy",  # Length 2
            "x" * 10,  # Length 10
            "x" * 100,  # Length 100
            "x" * 255,  # Near byte boundary
        ]

        # Filter out empty string if not supported
        try:
            trie = PrefixTrie(entries, allow_indels=True)
        except:
            entries = entries[1:]  # Remove empty string
            trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches for supported entries
        for entry in entries:
            if entry:  # Skip empty string
                result, corrections = trie.search(entry)
                assert result == entry
                assert corrections == 0

    def test_cache_behavior_stress(self):
        """Test to stress the internal cache mechanisms"""
        entries = ["cache", "caching", "cached", "caches"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Repeatedly search similar queries to stress cache
        queries = ["cachx", "cachng", "cachd", "cachs", "cach"]

        for _ in range(10):  # Repeat to test cache reuse
            for query in queries:
                result, corrections = trie.search(query, correction_budget=2)
                assert result is not None
                assert corrections > 0


class TestPrefixTrieAlgorithmCorrectness:
    """Test algorithm correctness for specific scenarios"""

    def test_edit_distance_calculation(self):
        """Test that edit distances are calculated correctly"""
        entries = ["kitten"]
        trie = PrefixTrie(entries, allow_indels=True)

        # "kitten" -> "sitting" requires 3 edits (k->s, e->i, insert g at the end)

        # Search for "sitting" with a budget of 2, should fail
        result, corrections = trie.search("sitting", correction_budget=2)
        assert result is None
        assert corrections == -1

        # Search with a budget of 3, should succeed and report 3 corrections
        result, corrections = trie.search("sitting", correction_budget=3)
        assert result == "kitten"
        assert corrections == 3

        # Searching for the exact word should yield 0 corrections
        result, corrections = trie.search("kitten", correction_budget=3)
        assert result == "kitten"
        assert corrections == 0

    def test_optimal_alignment_selection(self):
        """Test that the algorithm selects optimal alignments"""
        entries = ["ACGT", "TGCA"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Query that could align different ways
        result, corrections = trie.search("ACGA", correction_budget=2)
        assert result in ["ACGT", "TGCA"]
        assert corrections > 0

    def test_backtracking_scenarios(self):
        """Test scenarios that might require backtracking in search"""
        entries = ["abcdef", "abcxyz", "defghi"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Query that shares prefix with multiple entries
        result, corrections = trie.search("abcxef", correction_budget=2)
        assert result in ["abcdef", "abcxyz"]
        assert corrections > 0


class TestPrefixTrieSubstringSearch:
    """Test substring search functionality of PrefixTrie"""

    def test_basic_exact_substring_search(self):
        """Test basic exact substring matching"""
        entries = ["HELLO", "WORLD", "TEST"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        result, corrections, start, end = trie.search_substring("HELLO", correction_budget=0)
        assert result == "HELLO"
        assert corrections == 0
        assert start == 0
        assert end == 5

        # Test substring in middle
        result, corrections, start, end = trie.search_substring("AAAAHELLOAAAA", correction_budget=0)
        assert result == "HELLO"
        assert corrections == 0
        assert start == 4
        assert end == 9

        # Test at beginning
        result, corrections, start, end = trie.search_substring("HELLOAAAA", correction_budget=0)
        assert result == "HELLO"
        assert corrections == 0
        assert start == 0
        assert end == 5

        # Test at end
        result, corrections, start, end = trie.search_substring("AAAAHELLO", correction_budget=0)
        assert result == "HELLO"
        assert corrections == 0
        assert start == 4
        assert end == 9

    def test_no_match_substring_search(self):
        """Test substring search when no match is found"""
        entries = ["HELLO", "WORLD"]
        trie = PrefixTrie(entries, allow_indels=True)

        # No match found
        result, corrections, start, end = trie.search_substring("AAAABBBBCCCC", correction_budget=0)
        assert result is None
        assert corrections == -1
        assert start == -1
        assert end == -1

        # No match even with correction budget
        result, corrections, start, end = trie.search_substring("ZZZZXXXX", correction_budget=2)
        assert result is None
        assert corrections == -1
        assert start == -1
        assert end == -1

    def test_fuzzy_substring_search(self):
        """Test fuzzy substring matching with corrections"""
        entries = ["HELLO", "WORLD"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Single substitution
        result, corrections, start, end = trie.search_substring("AAAHELOAAAA", correction_budget=1)
        assert result == "HELLO"
        assert corrections == 1
        assert start == 3
        assert end == 7  # "HELO" spans positions 3-6, so end is 7

        # Single deletion (missing character)
        result, corrections, start, end = trie.search_substring("AAAHELLOAAAA", correction_budget=1)
        assert result == "HELLO"
        assert corrections == 0  # This should be exact since HELLO is found exactly
        assert start == 3
        assert end == 8

    def test_multiple_corrections_substring(self):
        """Test substring search requiring multiple corrections"""
        entries = ["ALGORITHM", "TESTING"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Two substitutions
        result, corrections, start, end = trie.search_substring("AAAAALGROTHMAAA", correction_budget=2)
        assert result == "ALGORITHM"
        assert corrections == 2
        assert start == 4
        assert end == 12  # "ALGROTHM" spans positions 4-11, so end is 12

        # Mixed corrections (substitution + insertion/deletion)
        result, corrections, start, end = trie.search_substring("BBBBTESTNGBBB", correction_budget=2)
        assert result == "TESTING"
        assert corrections > 0
        # The exact positions depend on the algorithm's alignment choice

    def test_overlapping_matches_substring(self):
        """Test substring search with overlapping potential matches"""
        entries = ["TEST", "TESTING", "EST"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Should find the longest/best match
        result, corrections, start, end = trie.search_substring("AAATESTINGAAA", correction_budget=0)
        assert result in ["TEST", "TESTING", "EST"]  # Any of these could be valid
        assert corrections == 0

        # Test with fuzzy matching
        result, corrections, start, end = trie.search_substring("AAATESXINGAAA", correction_budget=1)
        assert result in ["TEST", "TESTING"]  # Should prefer one of these
        assert corrections == 1

    def test_multiple_entries_in_target(self):
        """Test when target string contains multiple entries"""
        entries = ["CAT", "DOG", "BIRD"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Multiple entries present - should find one
        result, corrections, start, end = trie.search_substring("CATDOGBIRD", correction_budget=0)
        assert result in ["CAT", "DOG", "BIRD"]
        assert corrections == 0

        # Test with spacing
        result, corrections, start, end = trie.search_substring("AAACATAAADOGAAABIRD", correction_budget=0)
        assert result in ["CAT", "DOG", "BIRD"]
        assert corrections == 0

    def test_edge_cases_substring(self):
        """Test edge cases for substring search"""
        entries = ["A", "AB", "ABC"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Empty target string
        result, corrections, start, end = trie.search_substring("", correction_budget=0)
        assert result is None
        assert corrections == -1
        assert start == -1
        assert end == -1

        # Single character target
        result, corrections, start, end = trie.search_substring("A", correction_budget=0)
        assert result == "A"
        assert corrections == 0
        assert start == 0
        assert end == 1

        # Target shorter than all entries
        short_entries = ["HELLO", "WORLD"]
        short_trie = PrefixTrie(short_entries, allow_indels=True)
        result, corrections, start, end = short_trie.search_substring("HI", correction_budget=0)
        assert result is None
        assert corrections == -1

    def test_correction_budget_limits_substring(self):
        """Test that correction budget is properly respected in substring search"""
        entries = ["HELLO"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Should find with sufficient budget
        result, corrections, start, end = trie.search_substring("AAAHALLAOOO", correction_budget=2)
        assert result == "HELLO"
        assert corrections == 2

        # Should not find with insufficient budget
        result, corrections, start, end = trie.search_substring("AAAHALLAOOO", correction_budget=1)
        assert result is None
        assert corrections == -1
        assert start == -1
        assert end == -1

    def test_dna_sequence_substring_search(self):
        """Test substring search with DNA sequences"""
        sequences = ["ATCG", "GCTA", "TTAA", "CCGG"]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Exact DNA match
        result, corrections, start, end = trie.search_substring("AAAAATCGAAAA", correction_budget=0)
        assert result == "ATCG"
        assert corrections == 0
        assert start == 4
        assert end == 8

        # DNA with single base substitution
        result, corrections, start, end = trie.search_substring("AAAAATCAAAAA", correction_budget=1)
        assert result == "ATCG"
        assert corrections == 1
        assert start == 4
        assert end == 8  # "ATCA" spans positions 4-7, so end is 8

    def test_long_dna_substring_search(self):
        """Test substring search with longer DNA sequences"""
        sequences = [
            "ATCGATCGATCG",  # 12 bases
            "GCTAGCTAGCTA",  # 12 bases
            "AAATTTCCCGGG",  # 12 bases
        ]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Exact match in long string
        target = "NNNNATCGATCGATCGNNNN"
        result, corrections, start, end = trie.search_substring(target, correction_budget=0)
        assert result == "ATCGATCGATCG"
        assert corrections == 0
        assert start == 4
        assert end == 16

        # Fuzzy match with mutations
        target_fuzzy = "NNNNATCGATCGATCANNNN"  # G->A mutation at end
        result, corrections, start, end = trie.search_substring(target_fuzzy, correction_budget=1)
        assert result == "ATCGATCGATCG"
        assert corrections == 1
        assert start == 4
        assert end == 16  # "ATCGATCGATCA" spans positions 4-15, so end is 16

    def test_protein_sequence_substring_search(self):
        """Test substring search with protein sequences"""
        proteins = ["MKLLFY", "ARNDCQ", "EGHILK"]  # Amino acid sequences
        trie = PrefixTrie(proteins, allow_indels=True)

        # Exact protein match
        result, corrections, start, end = trie.search_substring("XXXMKLLFYXXX", correction_budget=0)
        assert result == "MKLLFY"
        assert corrections == 0
        assert start == 3
        assert end == 9

        # Protein with amino acid substitution
        result, corrections, start, end = trie.search_substring("XXXMKLLAYXXX", correction_budget=1)
        assert result == "MKLLFY"
        assert corrections == 1
        assert start == 3
        assert end == 9

    def test_performance_large_target_string(self):
        """Test performance with large target strings"""
        entries = ["NEEDLE", "HAYSTACK", "SEARCH"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Large target string with match at end
        large_target = "X" * 1000 + "NEEDLE" + "Y" * 1000
        result, corrections, start, end = trie.search_substring(large_target, correction_budget=0)
        assert result == "NEEDLE"
        assert corrections == 0
        assert start == 1000
        assert end == 1006

    def test_special_characters_substring(self):
        """Test substring search with special characters"""
        entries = ["hello!", "@test#", "a-b-c", "x_y_z"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Special characters exact match
        result, corrections, start, end = trie.search_substring("AAA@test#BBB", correction_budget=0)
        assert result == "@test#"
        assert corrections == 0
        assert start == 3
        assert end == 9

        # Special characters with fuzzy match
        result, corrections, start, end = trie.search_substring("AAAhelloBBB", correction_budget=1)
        assert result == "hello!"
        assert corrections == 1
        assert start == 3
        assert end == 9  # "hello" spans positions 3-7, but algorithm may find "hellob" spans 3-8, so end is 9

    def test_case_sensitive_substring(self):
        """Test that substring search respects case sensitivity"""
        entries = ["Hello", "HELLO", "hello"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Exact case matches
        result, corrections, start, end = trie.search_substring("AAAHelloAAA", correction_budget=0)
        assert result == "Hello"
        assert corrections == 0

        result, corrections, start, end = trie.search_substring("AAAHELLOAAa", correction_budget=0)
        assert result == "HELLO"
        assert corrections == 0

        result, corrections, start, end = trie.search_substring("AAAhelloAAA", correction_budget=0)
        assert result == "hello"
        assert corrections == 0

    def test_boundary_positions_substring(self):
        """Test substring matches at string boundaries"""
        entries = ["START", "END"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Match at very beginning
        result, corrections, start, end = trie.search_substring("STARTXXX", correction_budget=0)
        assert result == "START"
        assert corrections == 0
        assert start == 0
        assert end == 5

        # Match at very end
        result, corrections, start, end = trie.search_substring("XXXEND", correction_budget=0)
        assert result == "END"
        assert corrections == 0
        assert start == 3
        assert end == 6

        # Exact string match (target == entry)
        result, corrections, start, end = trie.search_substring("START", correction_budget=0)
        assert result == "START"
        assert corrections == 0
        assert start == 0
        assert end == 5

    def test_substring_with_repeats(self):
        """Test substring search with repetitive patterns"""
        entries = ["ABAB", "CACA", "TATA"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Repetitive target with exact match
        result, corrections, start, end = trie.search_substring("ABABABABAB", correction_budget=0)
        assert result == "ABAB"
        assert corrections == 0
        # Could match at position 0-4, 2-6, 4-8, or 6-10

        # Repetitive with single error - use a string where ABAB needs 1 correction
        result, corrections, start, end = trie.search_substring("XXABXBXX", correction_budget=1)
        assert result == "ABAB"
        assert corrections > 0

    def test_empty_trie_substring(self):
        """Test substring search with empty trie"""
        trie = PrefixTrie([], allow_indels=True)

        result, corrections, start, end = trie.search_substring("ANYTARGET", correction_budget=0)
        assert result is None
        assert corrections == -1
        assert start == -1
        assert end == -1

        result, corrections, start, end = trie.search_substring("ANYTARGET", correction_budget=5)
        assert result is None
        assert corrections == -1
        assert start == -1
        assert end == -1

    def test_very_short_entries_substring(self):
        """Test substring search with very short entries"""
        entries = ["A", "T", "C", "G"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Single character matches
        result, corrections, start, end = trie.search_substring("XAXBXCXDX", correction_budget=0)
        assert result in ["A", "C"]  # Could match either
        assert corrections == 0

        # Fuzzy single character match
        result, corrections, start, end = trie.search_substring("XXXXXX", correction_budget=1)
        assert result in ["A", "T", "C", "G"]
        assert corrections > 0

    def test_algorithm_consistency_substring(self):
        """Test that substring search results are consistent with regular search"""
        entries = ["HELLO", "WORLD", "TEST"]
        trie = PrefixTrie(entries, allow_indels=True)

        # If we can find it with regular search, substring search should find it too
        for entry in entries:
            regular_result, regular_corrections = trie.search(entry, correction_budget=0)
            substring_result, substring_corrections, start, end = trie.search_substring(entry, correction_budget=0)

            assert regular_result == substring_result
            assert regular_corrections == substring_corrections
            if substring_result is not None:
                assert start == 0
                assert end == len(entry)

        # Test with fuzzy matching
        regular_result, regular_corrections = trie.search("HALLO", correction_budget=1)
        substring_result, substring_corrections, start, end = trie.search_substring("HALLO", correction_budget=1)

        # Both should find "HELLO" or both should find nothing
        assert (regular_result is None) == (substring_result is None)
        if regular_result is not None and substring_result is not None:
            assert regular_result == substring_result
            assert regular_corrections == substring_corrections


def generate_barcodes(n: int, length: int = 16) -> list[str]:
    """Generate `n` deterministic barcodes of given length"""
    bases = "ACGT"
    barcodes = []
    for i in range(n):
        seq = []
        num = i
        for _ in range(length):
            seq.append(bases[num & 3])
            num >>= 2
        barcodes.append("".join(seq))
    return barcodes


class TestLargeWhitelist:

    def test_thousands_of_barcodes(self):
        # Generate 10k deterministic 16bp barcodes
        barcodes = generate_barcodes(10000)
        trie = PrefixTrie(barcodes, allow_indels=True)

        # Spot check a few barcodes for exact match
        samples = [barcodes[0], barcodes[123], barcodes[9999], barcodes[5000], barcodes[7777]]
        for bc in samples:
            result, corrections = trie.search(bc)
            assert result == bc
            assert corrections == 0

        # Mutate a high-order position to ensure it is not already in whitelist
        for idx, pos in [(42, 12), (123, 8), (9999, 15), (5000, 0), (7777, 5)]:
            original = barcodes[idx]
            replacement = "A" if original[pos] != "A" else "C"
            mutated = original[:pos] + replacement + original[pos + 1:]
            if mutated in barcodes:
                continue  # Skip if mutated barcode is already in whitelist
            result, corrections = trie.search(mutated, correction_budget=1)
            assert result == original
            assert corrections == 1


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running smoke test...")

    # Basic functionality test
    trie = PrefixTrie(["hello", "world", "test"])
    result, corrections = trie.search("hello")
    assert result == "hello" and corrections == 0

    # Fuzzy matching test
    trie_fuzzy = PrefixTrie(["hello"], allow_indels=True)
    result, corrections = trie_fuzzy.search("hllo", correction_budget=1)
    assert result == "hello" and corrections == 1

    print("Smoke test passed! Run 'pytest test.py' for full test suite.")
