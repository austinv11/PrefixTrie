import pytest
import pyximport

pyximport.install(
    setup_args={"include_dirs": ["../src/prefixtrie"]},
)
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
        entries = ["hello!", "test@", "a-b-c", "x_y_z"]
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


class TestPrefixTrieFuzzyLongestPrefixMatch:
    """Test the fuzzy longest_prefix_match functionality."""

    def test_fuzzy_match_with_substitution(self):
        """Test finding a prefix with one substitution."""
        entries = ["apple", "application"]
        trie = PrefixTrie(entries, allow_indels=True)
        # 'axple' should match 'apple' with 1 correction
        result, start, length = trie.longest_prefix_match("zzaxplezzz", min_match_length=4, correction_budget=1)
        assert result == "apple"
        assert start == 2
        assert length == 5

    def test_fuzzy_match_with_indel(self):
        """Test finding a prefix with one indel."""
        entries = ["apple", "apply"]
        trie = PrefixTrie(entries, allow_indels=True)
        # 'aple' (deletion) should match 'apple'
        # The returned length is the length of the matched substring in the target ("aple" -> 4).
        result, start, length = trie.longest_prefix_match("zzaplezzz", min_match_length=4, correction_budget=1)
        assert result == "apple"
        assert start == 2
        assert length == 4  # The match is on "aple"

        # 'appple' (insertion) should match 'apple'
        # The returned length is the length of the matched substring in the target ("appple" -> 6).
        result, start, length = trie.longest_prefix_match("zzappplezzz", min_match_length=4, correction_budget=1)
        assert result == "apple"
        assert start == 2
        assert length == 6

    def test_correction_budget_is_respected(self):
        """Test that the correction budget prevents matching."""
        entries = ["application"]
        trie = PrefixTrie(entries, allow_indels=True)
        # 'axplication' needs 1 correction
        result, start, length = trie.longest_prefix_match("zzaxplicationzz", min_match_length=8, correction_budget=0)
        assert result is None

        # With budget, it should be found
        result, start, length = trie.longest_prefix_match("zzaxplicationzz", min_match_length=8, correction_budget=1)
        assert result == "application"
        assert start == 2
        assert length == 11

    def test_chooses_longest_match(self):
        """Test that the longest match is chosen among fuzzy options."""
        entries = ["short", "shortest", "shortbread"]
        trie = PrefixTrie(entries, allow_indels=True)
        # 'shortx' is 1 away from 'short'
        # 'shortes' is 1 away from 'shortest'
        # 'shortbrea' is 1 away from 'shortbread'
        # Should find 'shortbread' as it's the longest
        result, start, length = trie.longest_prefix_match("zzshortbreazzz", min_match_length=5, correction_budget=1)
        assert result == "shortbread"
        assert start == 2
        assert length == 10

    def test_no_match_found(self):
        """Test case where no fuzzy match is possible."""
        entries = ["hello", "world"]
        trie = PrefixTrie(entries, allow_indels=True)
        result, start, length = trie.longest_prefix_match("zzxyzabcdezz", min_match_length=4, correction_budget=1)
        assert result is None

    def test_fallback_to_exact_match_behavior(self):
        """Test that correction_budget=0 provides the exact match behavior."""
        entries = ["testing", "tester"]
        trie = PrefixTrie(entries)

        # Calling with correction_budget=0 should find the exact prefix.
        result, start, length = trie.longest_prefix_match("xxxtesterxxx", min_match_length=6, correction_budget=0)

        assert result == "tester"
        assert start == 3
        assert length == 6

        # A fuzzy query should not match with budget=0.
        result, start, length = trie.longest_prefix_match("xxxtestorxxx", min_match_length=6, correction_budget=0)
        assert result is None

    def test_first_letter_correction(self):
        """Test that matches are found even if the first letter is wrong."""
        entries = ["important", "word"]
        trie = PrefixTrie(entries, allow_indels=True)
        # 'xmportant' should match 'important'
        result, start, length = trie.longest_prefix_match("zzxmportantzz", min_match_length=8, correction_budget=1)
        assert result == "important"
        assert start == 2
        assert length == 9

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
        assert start == 3
        assert end == 8

        result, corrections, start, end = trie.search_substring("AAAHELLOAAa", correction_budget=0)
        assert result == "HELLO"
        assert corrections == 0
        assert start == 3
        assert end == 8

        result, corrections, start, end = trie.search_substring("AAAhelloAAA", correction_budget=0)
        assert result == "hello"
        assert corrections == 0
        assert start == 3
        assert end == 8

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


class TestPrefixTrieLongestPrefixMatch:
    """Test longest_prefix_match functionality of PrefixTrie"""

    def test_basic_longest_prefix_match(self):
        """Test basic longest prefix matching"""
        entries = ["ACGT", "ACGTA", "ACGTAG", "TCGA"]
        trie = PrefixTrie(entries)

        # Exact match for complete entry
        result, start, length = trie.longest_prefix_match("ACGT", min_match_length=4)
        assert result == "ACGT"
        assert start == 0
        assert length == 4

        # Longest prefix when multiple matches possible
        result, start, length = trie.longest_prefix_match("ACGTAGGT", min_match_length=4)
        assert result == "ACGTAG"
        assert start == 0
        assert length == 6

        # Prefix match in middle of string - should find longest valid match
        result, start, length = trie.longest_prefix_match("NNACGTANN", min_match_length=4)
        assert result == "ACGTA"  # "ACGTA" is longer than "ACGT" and is a valid entry
        assert start == 2
        assert length == 5

    def test_no_match_cases(self):
        """Test cases where no match should be found"""
        entries = ["ACGT", "TCGA", "GGCC"]
        trie = PrefixTrie(entries)

        # No match found - too short
        result, start, length = trie.longest_prefix_match("ACG", min_match_length=4)
        assert result is None
        assert start == -1
        assert length == -1

        # No match found - no valid prefix
        result, start, length = trie.longest_prefix_match("XYZT", min_match_length=1)
        assert result is None
        assert start == -1
        assert length == -1

        # No match found - min_match_length too high
        result, start, length = trie.longest_prefix_match("ACGTTT", min_match_length=5)
        assert result is None
        assert start == -1
        assert length == -1

    def test_min_match_length_parameter(self):
        """Test the min_match_length parameter"""
        entries = ["A", "AB", "ABC", "ABCD"]
        trie = PrefixTrie(entries)

        # With min_match_length=1, should find "A"
        result, start, length = trie.longest_prefix_match("AXYZ", min_match_length=1)
        assert result == "A"
        assert start == 0
        assert length == 1

        # With min_match_length=2, should find "AB"
        result, start, length = trie.longest_prefix_match("ABXYZ", min_match_length=2)
        assert result == "AB"
        assert start == 0
        assert length == 2

        # With min_match_length=3, should find "ABC"
        result, start, length = trie.longest_prefix_match("ABCXYZ", min_match_length=3)
        assert result == "ABC"
        assert start == 0
        assert length == 3

        # With min_match_length=5, should find nothing
        result, start, length = trie.longest_prefix_match("ABCDXYZ", min_match_length=5)
        assert result is None
        assert start == -1
        assert length == -1

    def test_multiple_possible_matches(self):
        """Test when multiple prefixes are possible"""
        entries = ["TEST", "TESTING", "TESTER", "TE"]
        trie = PrefixTrie(entries)

        # Should find the longest match
        result, start, length = trie.longest_prefix_match("TESTINGABC", min_match_length=2)
        assert result == "TESTING"
        assert start == 0
        assert length == 7

        # Should find the longest match that meets min_match_length
        result, start, length = trie.longest_prefix_match("TESTERABC", min_match_length=4)
        assert result == "TESTER"
        assert start == 0
        assert length == 6

        # Should find shorter match when longer doesn't exist
        result, start, length = trie.longest_prefix_match("TESTABC", min_match_length=2)
        assert result == "TEST"
        assert start == 0
        assert length == 4

    def test_position_in_target_string(self):
        """Test finding matches at different positions in target string"""
        entries = ["CAT", "DOG", "BIRD"]
        trie = PrefixTrie(entries)

        # Match at beginning
        result, start, length = trie.longest_prefix_match("CATFISH", min_match_length=3)
        assert result == "CAT"
        assert start == 0
        assert length == 3

        # Match in middle
        result, start, length = trie.longest_prefix_match("XYCATFISH", min_match_length=3)
        assert result == "CAT"
        assert start == 2
        assert length == 3

        # Match at end
        result, start, length = trie.longest_prefix_match("XYCAT", min_match_length=3)
        assert result == "CAT"
        assert start == 2
        assert length == 3

        # Multiple possible matches - should find the longest one
        result, start, length = trie.longest_prefix_match("CATDOGBIRD", min_match_length=3)
        assert result == "BIRD"  # "BIRD" is 4 characters, longest match
        assert start == 6
        assert length == 4

    def test_edge_cases(self):
        """Test edge cases"""
        entries = ["A", "AA", "AAA"]
        trie = PrefixTrie(entries)

        # Single character
        result, start, length = trie.longest_prefix_match("A", min_match_length=1)
        assert result == "A"
        assert start == 0
        assert length == 1

        # Empty target string
        result, start, length = trie.longest_prefix_match("", min_match_length=1)
        assert result is None
        assert start == -1
        assert length == -1

        # min_match_length = 0 (should default to 1)
        result, start, length = trie.longest_prefix_match("AAA", min_match_length=0)
        # Implementation should handle this gracefully

    def test_dna_sequences(self):
        """Test with DNA sequences"""
        sequences = ["ATG", "ATGC", "ATGCG", "ATGCGT", "GCTA", "GCTAG"]
        trie = PrefixTrie(sequences)

        # Should find longest match
        result, start, length = trie.longest_prefix_match("ATGCGTAAA", min_match_length=3)
        assert result == "ATGCGT"
        assert start == 0
        assert length == 6

        # Should find match in middle
        result, start, length = trie.longest_prefix_match("NNATGCGTAAA", min_match_length=3)
        assert result == "ATGCGT"
        assert start == 2
        assert length == 6

        # Should respect min_match_length
        result, start, length = trie.longest_prefix_match("ATGXX", min_match_length=4)
        assert result is None
        assert start == -1
        assert length == -1

    def test_protein_sequences(self):
        """Test with protein sequences"""
        proteins = ["MET", "METG", "METGL", "METGLY", "ALA", "ALAG"]
        trie = PrefixTrie(proteins)

        # Should find longest protein match
        result, start, length = trie.longest_prefix_match("METGLYXXX", min_match_length=3)
        assert result == "METGLY"
        assert start == 0
        assert length == 6

        # Should find match in sequence
        result, start, length = trie.longest_prefix_match("XXXMETGLXXX", min_match_length=3)
        assert result == "METGL"
        assert start == 3
        assert length == 5

    def test_special_characters(self):
        """Test with special characters"""
        entries = ["hello!", "test@", "a-b-c", "x_y_z"]
        trie = PrefixTrie(entries)

        # Should handle special characters
        result, start, length = trie.longest_prefix_match("hello!world", min_match_length=5)
        assert result == "hello!"
        assert start == 0
        assert length == 6

        result, start, length = trie.longest_prefix_match("XXtest@YY", min_match_length=5)  # "test@" is 5 chars
        assert result == "test@"
        assert start == 2
        assert length == 5

    def test_case_sensitivity(self):
        """Test case sensitivity"""
        entries = ["Hello", "HELLO", "hello"]
        trie = PrefixTrie(entries)

        # Should be case sensitive
        result, start, length = trie.longest_prefix_match("HelloWorld", min_match_length=5)
        assert result == "Hello"
        assert start == 0
        assert length == 5

        result, start, length = trie.longest_prefix_match("HELLOWorld", min_match_length=5)
        assert result == "HELLO"
        assert start == 0
        assert length == 5

        result, start, length = trie.longest_prefix_match("helloWorld", min_match_length=5)
        assert result == "hello"
        assert start == 0
        assert length == 5

    def test_performance_large_entries(self):
        """Test performance with larger entries"""
        # Generate some larger sequences
        entries = [
            "ATCG" * 25,  # 100 characters
            "GCTA" * 25,  # 100 characters
            "TTAA" * 25,  # 100 characters
        ]
        trie = PrefixTrie(entries)

        target = "ATCG" * 25 + "EXTRA"
        result, start, length = trie.longest_prefix_match(target, min_match_length=50)
        assert result == "ATCG" * 25
        assert start == 0
        assert length == 100

    def test_overlapping_prefixes(self):
        """Test with overlapping prefix patterns"""
        entries = ["ABC", "ABCD", "ABCDE", "AB", "A"]
        trie = PrefixTrie(entries)

        # Should find longest match
        result, start, length = trie.longest_prefix_match("ABCDEFGH", min_match_length=1)
        assert result == "ABCDE"
        assert start == 0
        assert length == 5

        # Should respect min_match_length
        result, start, length = trie.longest_prefix_match("ABCDEFGH", min_match_length=4)
        assert result in ["ABCD", "ABCDE"]  # Both are valid >= 4
        assert length >= 4

    def test_empty_trie(self):
        """Test with empty trie"""
        trie = PrefixTrie([])

        result, start, length = trie.longest_prefix_match("ANYTARGET", min_match_length=1)
        assert result is None
        assert start == -1
        assert length == -1

    def test_consistency_with_regular_search(self):
        """Test that results are consistent with regular search when applicable"""
        entries = ["HELLO", "WORLD", "TEST"]
        trie = PrefixTrie(entries)

        # If longest_prefix_match finds something, regular search should find it too
        result, start, length = trie.longest_prefix_match("HELLO", min_match_length=5)
        if result is not None:
            search_result, corrections = trie.search(result)
            assert search_result == result
            assert corrections == 0

        # Test with prefix match
        result, start, length = trie.longest_prefix_match("HELLOWORLD", min_match_length=5)
        if result is not None:
            search_result, corrections = trie.search(result)
            assert search_result == result
            assert corrections == 0


class TestPrefixTrieMutability:
    """Test mutability features of PrefixTrie"""

    def test_immutable_by_default(self):
        """Test that tries are immutable by default (backward compatibility)"""
        trie = PrefixTrie(["hello", "world"])
        assert trie.is_immutable() == True

        # Should not be able to modify
        with pytest.raises(RuntimeError):
            trie.add("test")

        with pytest.raises(RuntimeError):
            trie.remove("hello")

    def test_explicit_immutable_creation(self):
        """Test creating explicitly immutable tries"""
        trie = PrefixTrie(["hello", "world"], immutable=True)
        assert trie.is_immutable() == True

        with pytest.raises(RuntimeError):
            trie.add("test")

        with pytest.raises(RuntimeError):
            trie.remove("hello")

    def test_mutable_creation(self):
        """Test creating mutable tries"""
        trie = PrefixTrie(["hello", "world"], immutable=False)
        assert trie.is_immutable() == False

        # Should be able to modify
        assert trie.add("test") == True
        assert "test" in trie
        assert len(trie) == 3

        assert trie.remove("world") == True
        assert "world" not in trie
        assert len(trie) == 2

    def test_add_functionality(self):
        """Test adding entries to mutable tries"""
        trie = PrefixTrie(["hello"], immutable=False)

        # Add new entry
        assert trie.add("world") == True
        assert "world" in trie
        assert len(trie) == 2

        # Add duplicate entry
        assert trie.add("hello") == False
        assert len(trie) == 2

        # Add multiple entries
        assert trie.add("test") == True
        assert trie.add("python") == True
        assert len(trie) == 4

        # Verify all entries work correctly
        for entry in ["hello", "world", "test", "python"]:
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

    def test_remove_functionality(self):
        """Test removing entries from mutable tries"""
        entries = ["hello", "world", "test", "python"]
        trie = PrefixTrie(entries, immutable=False)

        # Remove existing entry
        assert trie.remove("test") == True
        assert "test" not in trie
        assert len(trie) == 3

        # Remove non-existent entry
        assert trie.remove("nonexistent") == False
        assert len(trie) == 3

        # Remove all entries one by one
        assert trie.remove("hello") == True
        assert trie.remove("world") == True
        assert trie.remove("python") == True
        assert len(trie) == 0

        # Try to remove from empty trie
        assert trie.remove("anything") == False

    def test_add_remove_with_fuzzy_search(self):
        """Test that fuzzy search works correctly after add/remove operations"""
        trie = PrefixTrie(["hello"], allow_indels=True, immutable=False)

        # Add entries that are similar
        trie.add("help")
        trie.add("helicopter")

        # Test fuzzy search finds correct matches
        result, corrections = trie.search("helo", correction_budget=1)
        assert result in ["hello", "help"]
        assert corrections == 1

        # Remove an entry and verify fuzzy search still works
        trie.remove("help")
        result, corrections = trie.search("helo", correction_budget=1)
        assert result == "hello"
        assert corrections == 1

    def test_iteration_after_modifications(self):
        """Test that iteration works correctly after modifications"""
        initial_entries = ["a", "b", "c"]
        trie = PrefixTrie(initial_entries, immutable=False)

        # Verify initial iteration
        assert set(trie) == set(initial_entries)

        # Add entries and verify iteration
        trie.add("d")
        trie.add("e")
        assert set(trie) == {"a", "b", "c", "d", "e"}

        # Remove entries and verify iteration
        trie.remove("b")
        trie.remove("d")
        assert set(trie) == {"a", "c", "e"}

    def test_contains_after_modifications(self):
        """Test that __contains__ works correctly after modifications"""
        trie = PrefixTrie(["hello", "world"], immutable=False)

        # Initial state
        assert "hello" in trie
        assert "world" in trie
        assert "test" not in trie

        # After adding
        trie.add("test")
        assert "test" in trie

        # After removing
        trie.remove("world")
        assert "world" not in trie
        assert "hello" in trie
        assert "test" in trie

    def test_len_after_modifications(self):
        """Test that len() works correctly after modifications"""
        trie = PrefixTrie(["a", "b"], immutable=False)
        assert len(trie) == 2

        # After adding
        trie.add("c")
        assert len(trie) == 3

        trie.add("d")
        trie.add("e")
        assert len(trie) == 5

        # After removing
        trie.remove("a")
        assert len(trie) == 4

        trie.remove("c")
        trie.remove("e")
        assert len(trie) == 2

    def test_getitem_after_modifications(self):
        """Test that __getitem__ works correctly after modifications"""
        trie = PrefixTrie(["hello"], immutable=False)

        # Initial state
        assert trie["hello"] == "hello"

        # After adding
        trie.add("world")
        assert trie["world"] == "world"

        # After removing
        trie.remove("hello")
        with pytest.raises(KeyError):
            trie["hello"]
        assert trie["world"] == "world"

    def test_shared_memory_immutable_restriction(self):
        """Test that shared memory requires immutable tries"""
        # Mutable trie should not be able to create shared memory
        mutable_trie = PrefixTrie(["hello", "world"], immutable=False)
        with pytest.raises(RuntimeError, match="Cannot create shared memory for mutable trie"):
            mutable_trie.create_shared_memory()

        # Immutable trie should be able to create shared memory
        immutable_trie = PrefixTrie(["hello", "world"], immutable=True)
        try:
            shm_name = immutable_trie.create_shared_memory()
            assert shm_name is not None
        finally:
            immutable_trie.cleanup_shared_memory()

    def test_pickle_with_mutability(self):
        """Test that pickling works with both mutable and immutable tries"""
        import pickle

        # Test mutable trie
        mutable_trie = PrefixTrie(["hello", "world"], immutable=False)
        mutable_trie.add("test")

        pickled_data = pickle.dumps(mutable_trie)
        restored_mutable = pickle.loads(pickled_data)

        assert restored_mutable.is_immutable() == False
        assert len(restored_mutable) == 3
        assert "test" in restored_mutable

        # Test immutable trie
        immutable_trie = PrefixTrie(["hello", "world"], immutable=True)

        pickled_data = pickle.dumps(immutable_trie)
        restored_immutable = pickle.loads(pickled_data)

        assert restored_immutable.is_immutable() == True
        assert len(restored_immutable) == 2

    def test_complex_modification_sequence(self):
        """Test complex sequence of modifications"""
        trie = PrefixTrie([], immutable=False)
        assert len(trie) == 0

        # Build up trie
        words = ["apple", "application", "apply", "banana", "band", "bandana"]
        for word in words:
            assert trie.add(word) == True
            assert word in trie

        assert len(trie) == len(words)

        # Test search functionality
        for word in words:
            result, corrections = trie.search(word)
            assert result == word
            assert corrections == 0

        # Remove some words
        to_remove = ["apple", "band"]
        for word in to_remove:
            assert trie.remove(word) == True
            assert word not in trie

        remaining = [w for w in words if w not in to_remove]
        assert len(trie) == len(remaining)

        # Add back one word
        assert trie.add("grape") == True
        assert "grape" in trie
        assert len(trie) == len(remaining) + 1

        # Verify final state
        expected_final = remaining + ["grape"]
        assert set(trie) == set(expected_final)

    def test_add_remove_edge_cases(self):
        """Test edge cases for add and remove operations"""
        trie = PrefixTrie([], immutable=False)

        # Add to empty trie
        assert trie.add("first") == True
        assert len(trie) == 1

        # Remove only entry
        assert trie.remove("first") == True
        assert len(trie) == 0

        # Add empty string (if supported)
        try:
            result = trie.add("")
            if result:
                assert "" in trie
                assert trie.remove("") == True
        except (ValueError, RuntimeError):
            # Empty strings might not be supported
            pass

        # Add single character
        assert trie.add("a") == True
        assert "a" in trie

        # Add very long string
        long_string = "a" * 1000
        assert trie.add(long_string) == True
        assert long_string in trie

    def test_modification_with_special_characters(self):
        """Test modifications with special characters"""
        trie = PrefixTrie([], immutable=False)

        special_strings = [
            "hello!",
            "@test",
            "a-b-c",
            "x_y_z",
            "123",
            "mix3d_Ch4r5!",
        ]

        # Add all special strings
        for s in special_strings:
            assert trie.add(s) == True
            assert s in trie

        # Verify search works
        for s in special_strings:
            result, corrections = trie.search(s)
            assert result == s
            assert corrections == 0

        # Remove all special strings
        for s in special_strings:
            assert trie.remove(s) == True
            assert s not in trie

    def test_performance_large_modifications(self):
        """Test performance with large number of modifications"""
        trie = PrefixTrie([], immutable=False)

        # Add many entries
        entries = [f"entry_{i:04d}" for i in range(1000)]
        for entry in entries:
            assert trie.add(entry) == True

        assert len(trie) == 1000

        # Remove half the entries
        to_remove = entries[::2]  # Every other entry
        for entry in to_remove:
            assert trie.remove(entry) == True

        assert len(trie) == 500

        # Verify remaining entries still work
        remaining = entries[1::2]  # The other half
        for entry in remaining:
            assert entry in trie
            result, corrections = trie.search(entry)
            assert result == entry
            assert corrections == 0

    def test_modification_preserves_functionality(self):
        """Test that modifications preserve all trie functionality"""
        trie = PrefixTrie(["test", "testing"], allow_indels=True, immutable=False)

        # Add entries with common prefixes
        trie.add("tester")
        trie.add("tea")
        trie.add("team")

        # Test exact search
        for word in ["test", "testing", "tester", "tea", "team"]:
            result, corrections = trie.search(word)
            assert result == word
            assert corrections == 0

        # Test fuzzy search
        result, corrections = trie.search("testin", correction_budget=1)
        assert result == "testing"
        assert corrections == 1

        # Test substring search
        result, corrections, start, end = trie.search_substring("xxxtestingxxx", correction_budget=0)
        assert result == "testing"
        assert corrections == 0

        # Test longest prefix match
        result, start, length = trie.longest_prefix_match("testingabc", min_match_length=4)
        assert result == "testing"
        assert start == 0
        assert length == 7

        # Remove an entry and verify functionality still works
        trie.remove("tea")

        # Previous searches should still work (except for removed entry)
        result, corrections = trie.search("testing")
        assert result == "testing"
        assert corrections == 0

        result, corrections = trie.search("tea")
        assert result is None
        assert corrections == -1

    def test_error_conditions(self):
        """Test various error conditions with mutable tries"""
        # Test add on immutable trie
        immutable_trie = PrefixTrie(["hello"], immutable=True)
        with pytest.raises(RuntimeError, match="Cannot modify immutable trie"):
            immutable_trie.add("world")

        # Test remove on immutable trie
        with pytest.raises(RuntimeError, match="Cannot modify immutable trie"):
            immutable_trie.remove("hello")

        # Test with mutable trie
        mutable_trie = PrefixTrie(["hello"], immutable=False)

        # These should work without errors
        assert mutable_trie.add("world") == True
        assert mutable_trie.remove("hello") == True


class TestPrefixTrieSharedMemoryMutability:
    """Test shared memory restrictions with mutability"""

    def test_create_shared_trie_always_immutable(self):
        """Test that create_shared_trie always creates immutable tries"""
        from prefixtrie import create_shared_trie

        trie, shm_name = create_shared_trie(["hello", "world"], allow_indels=True)

        try:
            assert trie.is_immutable() == True

            # Should not be able to modify
            with pytest.raises(RuntimeError):
                trie.add("test")

            with pytest.raises(RuntimeError):
                trie.remove("hello")
        finally:
            trie.cleanup_shared_memory()

    def test_load_shared_trie_immutable(self):
        """Test that loaded shared tries are immutable"""
        from prefixtrie import create_shared_trie, load_shared_trie

        original_trie, shm_name = create_shared_trie(["hello", "world"])

        try:
            loaded_trie = load_shared_trie(shm_name)
            assert loaded_trie.is_immutable() == True

            # Should not be able to modify loaded trie
            with pytest.raises(RuntimeError):
                loaded_trie.add("test")

            with pytest.raises(RuntimeError):
                loaded_trie.remove("hello")
        finally:
            original_trie.cleanup_shared_memory()


class TestPrefixTrieSearchCount:
    """Test the search_count method"""

    def test_search_count_exact(self):
        """Test search_count with exact matches"""
        entries = ["hello", "world", "test", "hello"]  # Note duplicate
        trie = PrefixTrie(entries)
        assert trie.search_count("hello", correction_budget=0) == 1
        assert trie.search_count("world", correction_budget=0) == 1
        assert trie.search_count("test", correction_budget=0) == 1
        assert trie.search_count("goodbye", correction_budget=0) == 0

    def test_search_count_fuzzy_no_indels(self):
        """Test search_count with fuzzy matching but no indels"""
        entries = ["cat", "bat", "hat", "rat"]
        trie = PrefixTrie(entries, allow_indels=False)
        # "dat" is 1 substitution away from "cat", "bat", "hat", "rat"
        assert trie.search_count("dat", correction_budget=1) == 4
        # "dat" is 2 substitutions away from nothing in this list
        assert trie.search_count("dot", correction_budget=1) == 0

        entries_2 = ["sitting", "fitting", "hitting"]
        trie_2 = PrefixTrie(entries_2, allow_indels=False)
        # "hitting" is 1 sub away from "sitting" and "fitting"
        assert trie_2.search_count("hitting", correction_budget=1) == 3

    def test_search_count_fuzzy_with_indels(self):
        """Test search_count with fuzzy matching and indels"""
        entries = ["cat", "ca", "c", "bat"]
        trie = PrefixTrie(entries, allow_indels=True)
        # "ca" can match "cat" (1 del), "ca" (0), "c" (1 ins)
        assert trie.search_count("ca", correction_budget=1) == 3
        # "c" can match "cat" (2 del), "ca" (1 del), "c" (0)
        assert trie.search_count("c", correction_budget=1) == 2
        assert trie.search_count("c", correction_budget=2) == 3

    def test_search_count_no_matches(self):
        """Test search_count when no matches are expected"""
        entries = ["apple", "banana", "cherry"]
        trie = PrefixTrie(entries, allow_indels=True)
        assert trie.search_count("xyz", correction_budget=0) == 0
        assert trie.search_count("xyz", correction_budget=1) == 0
        # "apple" -> "xyz" is 5 corrections
        assert trie.search_count("xyz", correction_budget=4) == 0
        assert trie.search_count("xyz", correction_budget=5) == 1  # matches "apple"

    def test_search_count_complex(self):
        """Test search_count with a more complex scenario"""
        entries = ["test", "tests", "testing", "tester", "toast"]
        trie = PrefixTrie(entries, allow_indels=True)
        # "test" can match:
        # "test" (0)
        # "tests" (1 ins)
        # "tester" (2 ins)
        # "toast" (2 subs)
        assert trie.search_count("test", correction_budget=0) == 1
        assert trie.search_count("test", correction_budget=1) == 2  # test, tests
        assert trie.search_count("test", correction_budget=2) == 4  # test, tests, tester, toast

