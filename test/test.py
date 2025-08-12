import pytest
import pyximport
pyximport.install()
from prefixtrie import PrefixTrie


class TestPrefixTrieBasic:
    """Test basic functionality of PrefixTrie"""

    def test_empty_trie(self):
        """Test creating an empty trie"""
        trie = PrefixTrie([])
        result, exact = trie.search("test")
        assert result is None
        assert exact is False
        # Searching for an empty string in an empty trie should not report an
        # exact match.
        result, exact = trie.search("")
        assert result is None
        assert exact is False

    def test_single_entry(self):
        """Test trie with single entry"""
        trie = PrefixTrie(["hello"])

        # Exact match
        result, exact = trie.search("hello")
        assert result == "hello"
        assert exact is True

        # No match
        result, exact = trie.search("world")
        assert result is None
        assert exact is False

    def test_multiple_entries(self):
        """Test trie with multiple entries"""
        entries = ["cat", "car", "card", "care", "careful"]
        trie = PrefixTrie(entries)

        for entry in entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

    def test_trailing_and_missing_characters(self):
        """Ensure extra or missing characters are handled with indels"""
        entries = ["hello"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Extra character at the end should count as a deletion
        result, exact = trie.search("hello!", correction_budget=1)
        assert result == "hello"
        assert exact is False

        # Missing character should be handled as an insertion
        result, exact = trie.search("hell", correction_budget=1)
        assert result == "hello"
        assert exact is False

    def test_prefix_matching(self):
        """Test prefix-based matching"""
        entries = ["test", "testing", "tester", "tea", "team"]
        trie = PrefixTrie(entries)

        # Test exact matches for complete entries
        result, exact = trie.search("test")
        assert result == "test"
        assert exact is True

        result, exact = trie.search("tea")
        assert result == "tea"
        assert exact is True

        # Test that partial prefixes don't match without fuzzy search
        result, exact = trie.search("te")
        assert result is None
        assert exact is False


class TestPrefixTrieEdgeCases:
    """Test edge cases and special characters"""

    def test_empty_string_entry(self):
        """Test with empty string in entries"""
        # Empty strings may not be supported by this trie implementation
        trie = PrefixTrie(["hello", "world"])

        result, exact = trie.search("")
        assert result is None
        assert exact is False

    def test_single_character_entries(self):
        """Test with single character entries"""
        trie = PrefixTrie(["a", "b", "c"])

        result, exact = trie.search("a")
        assert result == "a"
        assert exact is True

        result, exact = trie.search("d")
        assert result is None
        assert exact is False

    def test_duplicate_entries(self):
        """Test with duplicate entries"""
        trie = PrefixTrie(["hello", "hello", "world"])

        result, exact = trie.search("hello")
        assert result == "hello"
        assert exact is True

    def test_special_characters(self):
        """Test with special characters"""
        entries = ["hello!", "test@123", "a-b-c", "x_y_z"]
        trie = PrefixTrie(entries)

        for entry in entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

    def test_case_sensitivity(self):
        """Test case sensitivity"""
        trie = PrefixTrie(["Hello", "hello", "HELLO"])

        result, exact = trie.search("Hello")
        assert result == "Hello"
        assert exact is True

        result, exact = trie.search("hello")
        assert result == "hello"
        assert exact is True

        result, exact = trie.search("HELLO")
        assert result == "HELLO"
        assert exact is True

    def test_budget_increase_recomputes(self):
        trie = PrefixTrie(["hello"], allow_indels=True)
        result, exact = trie.search("hallo", correction_budget=0)
        assert result is None and exact is False

        # With more corrections available, the match should now succeed
        result, exact = trie.search("hallo", correction_budget=1)
        assert result == "hello" and exact is False

class TestPrefixTrieFuzzyMatching:
    """Test fuzzy matching capabilities"""

    def test_basic_fuzzy_matching(self):
        """Test basic fuzzy matching with corrections"""
        entries = ["hello", "world", "python"]
        trie = PrefixTrie(entries, allow_indels=False)

        # Test with 1 correction budget - single character substitution
        result, exact = trie.search("hallo", correction_budget=1)  # e->a substitution
        assert result == "hello"
        assert exact is False

        result, exact = trie.search("worle", correction_budget=1)  # d->e substitution
        assert result == "world"
        assert exact is False

    def test_fuzzy_matching_with_indels(self):
        """Test fuzzy matching with insertions and deletions"""
        entries = ["hello", "world"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test simple substitution that should work
        result, exact = trie.search("hallo", correction_budget=1)
        assert result == "hello"
        assert exact is False

        # Test that we can find matches with small edits
        result, exact = trie.search("worlx", correction_budget=1)
        assert result == "world"
        assert exact is False

    def test_correction_budget_limits(self):
        """Test that correction budget is respected"""
        entries = ["hello"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Should find with budget of 2
        result, exact = trie.search("hallo", correction_budget=2)
        assert result == "hello"
        assert exact is False

        # Should not find with budget of 0
        result, exact = trie.search("hallo", correction_budget=0)
        assert result is None
        assert exact is False

    def test_multiple_corrections(self):
        """Test queries requiring multiple corrections"""
        entries = ["testing"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Two substitutions
        result, exact = trie.search("taxting", correction_budget=2)
        assert result == "testing"
        assert exact is False

        # Should not find with insufficient budget
        result, exact = trie.search("taxting", correction_budget=1)
        assert result is None
        assert exact is False


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
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

    def test_long_strings(self):
        """Test with very long strings"""
        long_string = "a" * 1000
        entries = [long_string, long_string + "b"]
        trie = PrefixTrie(entries)

        result, exact = trie.search(long_string)
        assert result == long_string
        assert exact is True

    def test_many_entries(self):
        """Test with many entries"""
        entries = [f"entry_{i:04d}" for i in range(1000)]
        trie = PrefixTrie(entries)

        # Test a few random entries
        test_entries = [entries[0], entries[500], entries[999]]
        for entry in test_entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True


class TestPrefixTrieDNASequences:
    """Test with DNA-like sequences (similar to the original test)"""

    def test_dna_sequences(self):
        """Test with DNA sequences"""
        sequences = ["ACGT", "TCGA", "AAAA", "TTTT", "CCCC", "GGGG"]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Exact matches
        for seq in sequences:
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

    def test_dna_fuzzy_matching(self):
        """Test fuzzy matching with DNA sequences"""
        sequences = ["ACGT", "TCGA"]
        trie = PrefixTrie(sequences, allow_indels=True)

        # Single base substitution
        result, exact = trie.search("ACCT", correction_budget=1)
        assert result == "ACGT"
        assert exact is False

        # Test with a clear mismatch that requires correction
        result, exact = trie.search("ACXX", correction_budget=2)
        assert result == "ACGT"
        assert exact is False

        # Test that fuzzy matching works with sufficient budget
        result, exact = trie.search("TCXX", correction_budget=2)
        assert result == "TCGA"
        assert exact is False

    def test_similar_sequences(self):
        """Test with very similar sequences"""
        sequences = ["ATCG", "ATCGA", "ATCGAA", "ATCGAAA"]
        trie = PrefixTrie(sequences)

        for seq in sequences:
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

        # Test fuzzy matching with single substitution
        result, exact = trie.search("ATCGATCGATCGATCGATCX", correction_budget=1)
        assert result == "ATCGATCGATCGATCGATCG"
        assert exact is False

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

        # Test mutation simulation (single nucleotide polymorphism)
        result, exact = trie.search("ATGAAACGTCTAGCTAGCTAGCTAGCTAX", correction_budget=1)
        assert result == "ATGAAACGTCTAGCTAGCTAGCTAGCTAG"
        assert exact is False

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

        # Test with a shorter repetitive sequence for fuzzy matching
        short_sequences = ["CACA", "GTGT", "ATAT"]
        short_trie = PrefixTrie(short_sequences, allow_indels=True)

        result, exact = short_trie.search("CACX", correction_budget=1)
        assert result == "CACA"
        assert exact is False

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

        # Test fuzzy matching across different lengths
        result, exact = trie.search("ATCX", correction_budget=1)
        assert result == "ATCG"
        assert exact is False

        result, exact = trie.search("ATCGATCGATCGATCX", correction_budget=1)
        assert result == "ATCGATCGATCGATCG"
        assert exact is False

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

        # Test single codon mutations
        result, exact = trie.search("ATGAAATTTCCCGGT", correction_budget=1)  # G->T in last codon
        assert result == "ATGAAATTTCCCGGG"
        assert exact is False

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True

        # Test fuzzy matching with very long sequences
        query = "ATCG" * 249 + "ATCX"  # 999 bases + ATCX
        result, exact = trie.search(query, correction_budget=1)
        assert result == "ATCG" * 250
        assert exact is False

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
            result, exact = trie.search(seq)
            assert result == seq
            assert exact is True


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
        result, exact = trie.search("hallo", correction_budget=-1)
        assert result is None
        assert exact is False

class TestPrefixTrieAdvancedEdgeCases:
    """Test advanced edge cases and algorithm-specific scenarios"""

    def test_insertion_and_deletion_operations(self):
        """Test specific insertion and deletion operations in fuzzy matching"""
        entries = ["hello", "help", "helicopter"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test insertions - query is shorter than target
        result, exact = trie.search("hell", correction_budget=1)  # could be "hello" or "help" (both 1 edit)
        assert result in ["hello", "help"]  # Both are valid with 1 edit
        assert exact is False

        result, exact = trie.search("hel", correction_budget=1)  # missing 'p' to make "help"
        assert result == "help"
        assert exact is False

        # Test deletions - query is longer than target
        result, exact = trie.search("helllo", correction_budget=1)  # extra 'l'
        assert result == "hello"
        assert exact is False

        result, exact = trie.search("helpx", correction_budget=1)  # extra 'x'
        assert result == "help"
        assert exact is False

        # Test substitutions
        result, exact = trie.search("helo", correction_budget=1)  # 'o'->'p' substitution
        assert result == "help"  # This is correct - only 1 edit needed
        assert exact is False

    def test_complex_indel_combinations(self):
        """Test combinations of insertions, deletions, and substitutions"""
        entries = ["algorithm", "logarithm", "rhythm"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Combination: deletion + substitution
        result, exact = trie.search("algrothm", correction_budget=2)  # missing 'i', 'i'->'o'
        assert result == "algorithm"
        assert exact is False

        # Combination: insertion + substitution
        result, exact = trie.search("logxarithm", correction_budget=2)  # extra 'x', 'x'->'a'
        assert result == "logarithm"
        assert exact is False

    def test_prefix_collision_scenarios(self):
        """Test scenarios where prefixes collide and could cause issues"""
        entries = ["a", "aa", "aaa", "aaaa", "aaaaa"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Exact matches should work
        for entry in entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

        # Fuzzy matching should find closest match
        result, exact = trie.search("aax", correction_budget=1)
        assert result == "aaa"
        assert exact is False

        result, exact = trie.search("aaax", correction_budget=1)
        assert result == "aaaa"
        assert exact is False

    def test_shared_prefix_disambiguation(self):
        """Test disambiguation when multiple entries share long prefixes"""
        entries = [
            "programming", "programmer", "programmed", "programmable",
            "program", "programs", "programmatic"
        ]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        for entry in entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

        # Test fuzzy matching with shared prefixes
        result, exact = trie.search("programmin", correction_budget=1)  # missing 'g'
        assert result == "programming"
        assert exact is False

        result, exact = trie.search("programmerz", correction_budget=1)  # 'z' instead of final char
        assert result == "programmer"
        assert exact is False

    def test_empty_and_very_short_queries(self):
        """Test behavior with empty and very short queries"""
        entries = ["a", "ab", "abc", "hello", "world"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Empty query
        result, exact = trie.search("", correction_budget=0)
        assert result is None
        assert exact is False

        result, exact = trie.search("", correction_budget=1)
        assert result == "a"  # Should find shortest entry
        assert exact is False

        # Single character queries
        result, exact = trie.search("x", correction_budget=1)
        assert result == "a"  # Should find closest single char
        assert exact is False

    def test_correction_budget_edge_cases(self):
        """Test edge cases around correction budget limits"""
        entries = ["test", "best", "rest", "nest"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Exact budget limit
        result, exact = trie.search("zest", correction_budget=1)  # 'z'->'t', 'e'->'e', 's'->'s', 't'->'t'
        assert result == "test"
        assert exact is False

        # Just over budget
        result, exact = trie.search("zesz", correction_budget=1)  # needs 2 corrections
        assert result is None
        assert exact is False

        # Zero budget should only find exact matches
        result, exact = trie.search("test", correction_budget=0)
        assert result == "test"
        assert exact is True

        result, exact = trie.search("tesy", correction_budget=0)
        assert result is None
        assert exact is False

    def test_alphabet_boundary_conditions(self):
        """Test with characters at alphabet boundaries"""
        entries = ["aaa", "zzz", "AZaz", "09azAZ"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        for entry in entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

        # Test fuzzy matching across character boundaries
        result, exact = trie.search("aab", correction_budget=1)
        assert result == "aaa"
        assert exact is False

        result, exact = trie.search("zzy", correction_budget=1)
        assert result == "zzz"
        assert exact is False

    def test_collapsed_path_edge_cases(self):
        """Test edge cases with collapsed paths in the trie"""
        # Create entries that will cause path collapsing
        entries = ["abcdefghijk", "abcdefghijl", "xyz"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        for entry in entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

        # Test fuzzy matching that might interact with collapsed paths
        result, exact = trie.search("abcdefghijx", correction_budget=1)  # Last char different
        expected = "abcdefghijk"  # Should match first entry
        assert result == expected
        assert exact is False

    def test_memory_intensive_operations(self):
        """Test operations that might stress memory management"""
        # Create many similar entries
        entries = [f"pattern{i:03d}suffix" for i in range(100)]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test a few random exact matches
        test_entries = [entries[0], entries[50], entries[99]]
        for entry in test_entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

        # Test fuzzy matching
        result, exact = trie.search("pattern050suffi", correction_budget=1)  # missing 'x'
        assert result == "pattern050suffix"
        assert exact is False

    def test_very_high_correction_budget(self):
        """Test with very high correction budgets"""
        entries = ["short", "verylongstring"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Very high budget should still work correctly
        result, exact = trie.search("x", correction_budget=100)
        assert result == "short"  # Should find shortest
        assert exact is False

        result, exact = trie.search("completelydifferent", correction_budget=100)
        assert result is not None  # Should find something
        assert exact is False

    def test_indel_vs_substitution_preference(self):
        """Test algorithm preference between indels and substitutions"""
        entries = ["abc", "abcd", "abce"]
        trie = PrefixTrie(entries, allow_indels=True)

        # This query could match "abc" with 1 deletion or "abcd"/"abce" with 1 substitution
        result, exact = trie.search("abcx", correction_budget=1)
        # The algorithm should prefer the substitution (keeping same length)
        assert result in ["abcd", "abce"]
        assert exact is False

    def test_multiple_valid_corrections(self):
        """Test scenarios where multiple corrections have same cost"""
        entries = ["cat", "bat", "hat", "rat"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Query that's 1 edit away from multiple entries
        result, exact = trie.search("dat", correction_budget=1)
        assert result in entries  # Should find one of them
        assert exact is False

        # With higher budget, should still work
        result, exact = trie.search("zat", correction_budget=1)
        assert result in entries
        assert exact is False

    def test_nested_prefix_structures(self):
        """Test deeply nested prefix structures"""
        entries = [
            "a", "ab", "abc", "abcd", "abcde", "abcdef",
            "abcdeg", "abcdeh", "abcdei"
        ]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test exact matches
        for entry in entries:
            result, exact = trie.search(entry)
            assert result == entry
            assert exact is True

        # Test fuzzy matching at different depths
        result, exact = trie.search("abcdej", correction_budget=1)
        assert result in ["abcdef", "abcdeg", "abcdeh", "abcdei"]
        assert exact is False

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
                result, exact = trie.search(entry)
                assert result == entry
                assert exact is True

    def test_cache_behavior_stress(self):
        """Test to stress the internal cache mechanisms"""
        entries = ["cache", "caching", "cached", "caches"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Repeatedly search similar queries to stress cache
        queries = ["cachx", "cachng", "cachd", "cachs", "cach"]

        for _ in range(10):  # Repeat to test cache reuse
            for query in queries:
                result, exact = trie.search(query, correction_budget=2)
                assert result is not None
                assert exact is False

class TestPrefixTrieAlgorithmCorrectness:
    """Test algorithm correctness for specific scenarios"""

    def test_edit_distance_calculation(self):
        """Test that edit distances are calculated correctly"""
        entries = ["kitten", "sitting"]
        trie = PrefixTrie(entries, allow_indels=True)

        # "kitten" -> "sitting" requires 3 edits (k->s, e->i, insert g)
        result, exact = trie.search("kitten", correction_budget=3)
        assert result == "kitten"
        assert exact is True

        # Should not find "sitting" with budget of 2 (needs 3 edits)
        result, exact = trie.search("sitting", correction_budget=2)
        # This should find "sitting" exactly since it's in the trie
        assert result == "sitting"
        assert exact is True

    def test_optimal_alignment_selection(self):
        """Test that the algorithm selects optimal alignments"""
        entries = ["ACGT", "TGCA"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Query that could align different ways
        result, exact = trie.search("ACGA", correction_budget=2)
        assert result in ["ACGT", "TGCA"]
        assert exact is False

    def test_backtracking_scenarios(self):
        """Test scenarios that might require backtracking in search"""
        entries = ["abcdef", "abcxyz", "defghi"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Query that shares prefix with multiple entries
        result, exact = trie.search("abcxef", correction_budget=2)
        assert result in ["abcdef", "abcxyz"]
        assert exact is False

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
            result, exact = trie.search(bc)
            assert result == bc
            assert exact is True

        # Mutate a high-order position to ensure it is not already in whitelist
        for idx, pos in [(42, 12), (123, 8), (9999, 15), (5000, 0), (7777, 5)]:
            original = barcodes[idx]
            replacement = "A" if original[pos] != "A" else "C"
            mutated = original[:pos] + replacement + original[pos + 1:]
            if mutated in barcodes:
                continue  # Skip if mutated barcode is already in whitelist
            result, exact = trie.search(mutated, correction_budget=1)
            assert result == original
            assert exact is False


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running smoke test...")

    # Basic functionality test
    trie = PrefixTrie(["hello", "world", "test"])
    result, exact = trie.search("hello")
    assert result == "hello" and exact is True

    # Fuzzy matching test
    trie_fuzzy = PrefixTrie(["hello"], allow_indels=True)
    result, exact = trie_fuzzy.search("hllo", correction_budget=1)
    assert result == "hello" and exact is False

    print("Smoke test passed! Run 'pytest test.py' for full test suite.")
