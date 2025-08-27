import time
import statistics
import random
import string
import pytest
import pyximport
pyximport.install()
from prefixtrie import PrefixTrie

try:
    import rapidfuzz
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    pytest.skip("rapidfuzz not available", allow_module_level=True)


def generate_random_strings(n: int, length: int = 10, alphabet: str = None) -> list[str]:
    """Generate n random strings of given length"""
    if alphabet is None:
        alphabet = string.ascii_lowercase

    strings = []
    for _ in range(n):
        s = ''.join(random.choice(alphabet) for _ in range(length))
        strings.append(s)
    return strings


def generate_dna_sequences(n: int, length: int = 20) -> list[str]:
    """Generate n random DNA sequences"""
    return generate_random_strings(n, length, "ATCG")


def generate_protein_sequences(n: int, length: int = 30) -> list[str]:
    """Generate n random protein sequences using 20 amino acid alphabet"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return generate_random_strings(n, length, amino_acids)


def generate_realistic_words(n: int) -> list[str]:
    """Generate realistic-looking English words"""
    prefixes = ["pre", "un", "re", "in", "dis", "mis", "over", "under", "out", "up"]
    roots = ["test", "work", "play", "run", "jump", "walk", "talk", "read", "write", "sing",
             "dance", "cook", "clean", "build", "fix", "make", "take", "give", "find", "help"]
    suffixes = ["ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment", "able"]

    words = []
    for _ in range(n):
        if random.random() < 0.3:  # 30% chance for prefix
            word = random.choice(prefixes)
        else:
            word = ""

        word += random.choice(roots)

        if random.random() < 0.4:  # 40% chance for suffix
            word += random.choice(suffixes)

        words.append(word)

    return words


def generate_hierarchical_strings(n: int, levels: int = 3) -> list[str]:
    """Generate hierarchical strings like file paths or taxonomies"""
    level_names = [
        ["sys", "usr", "var", "home", "opt", "tmp"],
        ["bin", "lib", "src", "data", "config", "cache"],
        ["main", "test", "util", "core", "api", "ui"],
        ["file", "module", "class", "func", "var", "const"]
    ]

    strings = []
    for _ in range(n):
        parts = []
        for level in range(levels):
            if level < len(level_names):
                parts.append(random.choice(level_names[level]))
            else:
                parts.append(f"item{random.randint(1000, 9999)}")
        strings.append("/".join(parts))

    return strings


def generate_queries_with_errors(entries: list[str], error_rate: float = 0.1) -> list[str]:
    """Generate queries by introducing errors into existing entries"""
    queries = []
    alphabet = set(''.join(entries))
    alphabet = list(alphabet) if alphabet else list(string.ascii_lowercase)

    for entry in entries:
        if random.random() < 0.5:  # 50% chance to add errors
            # Introduce 1-2 errors
            query = list(entry)
            num_errors = random.randint(1, min(2, len(entry)))

            for _ in range(num_errors):
                if not query:
                    break

                error_type = random.choice(['substitute', 'insert', 'delete'])
                pos = random.randint(0, len(query) - 1) if query else 0

                if error_type == 'substitute' and pos < len(query):
                    query[pos] = random.choice(alphabet)
                elif error_type == 'insert':
                    query.insert(pos, random.choice(alphabet))
                elif error_type == 'delete' and query:
                    query.pop(pos)

            queries.append(''.join(query))
        else:
            # Keep some exact matches
            queries.append(entry)

    return queries


def validate_trie_consistency(entries: list[str], trie_results: list[tuple], test_name: str = ""):
    """Validate that trie results are consistent with expected behavior"""
    print(f"\n  Validating consistency for {test_name}...")

    entries_set = set(entries)
    inconsistencies = []

    for i, (result, exact) in enumerate(trie_results):
        if result is not None:
            # If result is found, it should be in the original entries
            if result not in entries_set:
                inconsistencies.append(f"Index {i}: Found '{result}' not in original entries")

            # If exact is True, the result should match a query exactly
            # Note: This is harder to validate without the original query

        # Additional consistency checks can be added here

    if inconsistencies:
        print(f"  WARNING: Found {len(inconsistencies)} inconsistencies:")
        for inc in inconsistencies[:5]:  # Show first 5
            print(f"    {inc}")
        if len(inconsistencies) > 5:
            print(f"    ... and {len(inconsistencies) - 5} more")
    else:
        print(f"  ✓ No inconsistencies found")

    return len(inconsistencies) == 0


def time_function(func, *args, **kwargs):
    """Time a function execution"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def benchmark_prefixtrie_exact(entries: list[str], queries: list[str]) -> list:
    """Benchmark PrefixTrie for exact matching"""
    trie = PrefixTrie(entries, allow_indels=False)

    results = []
    for query in queries:
        result, exact = trie.search(query)
        results.append((result, exact))

    return results


def benchmark_prefixtrie_fuzzy(entries: list[str], queries: list[str], budget: int = 2) -> list:
    """Benchmark PrefixTrie for fuzzy matching"""
    trie = PrefixTrie(entries, allow_indels=True)

    results = []
    for query in queries:
        result, exact = trie.search(query, correction_budget=budget)
        results.append((result, exact))

    return results


def benchmark_rapidfuzz_exact(entries: list[str], queries: list[str]) -> list:
    """Benchmark rapidfuzz for exact matching"""
    entries_set = set(entries)

    results = []
    for query in queries:
        if query in entries_set:
            results.append((query, True))
        else:
            results.append((None, False))

    return results


def benchmark_rapidfuzz_fuzzy(entries: list[str], queries: list[str], score_cutoff: int = 80) -> list:
    """Benchmark rapidfuzz for fuzzy matching"""
    results = []
    for query in queries:
        # Use extractOne to find best match
        match = process.extractOne(query, entries, score_cutoff=score_cutoff)
        if match:
            results.append((match[0], match[1] == 100))  # exact if score is 100
        else:
            results.append((None, False))

    return results


def run_benchmark_suite(name: str, entries: list[str], queries: list[str], num_runs: int = 5):
    """Run a complete benchmark suite"""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"Entries: {len(entries)}, Queries: {len(queries)}")
    print(f"{'='*60}")

    # Exact matching benchmarks
    print("\nEXACT MATCHING:")
    print("-" * 40)

    prefixtrie_exact_times = []
    rapidfuzz_exact_times = []

    # Store results for consistency checking
    pt_exact_results = None
    rf_exact_results = None

    for i in range(num_runs):
        # PrefixTrie exact
        results, time_taken = time_function(benchmark_prefixtrie_exact, entries, queries)
        prefixtrie_exact_times.append(time_taken)
        if pt_exact_results is None:
            pt_exact_results = results

        # RapidFuzz exact
        results, time_taken = time_function(benchmark_rapidfuzz_exact, entries, queries)
        rapidfuzz_exact_times.append(time_taken)
        if rf_exact_results is None:
            rf_exact_results = results

    # Validate consistency
    validate_trie_consistency(entries, pt_exact_results, f"{name} - PrefixTrie Exact")
    validate_trie_consistency(entries, rf_exact_results, f"{name} - RapidFuzz Exact")

    pt_exact_avg = statistics.mean(prefixtrie_exact_times)
    pt_exact_std = statistics.stdev(prefixtrie_exact_times) if len(prefixtrie_exact_times) > 1 else 0
    rf_exact_avg = statistics.mean(rapidfuzz_exact_times)
    rf_exact_std = statistics.stdev(rapidfuzz_exact_times) if len(rapidfuzz_exact_times) > 1 else 0

    print(f"PrefixTrie:  {pt_exact_avg:.4f}s ± {pt_exact_std:.4f}s")
    print(f"RapidFuzz:   {rf_exact_avg:.4f}s ± {rf_exact_std:.4f}s")
    print(f"Speedup:     {rf_exact_avg/pt_exact_avg:.2f}x" if pt_exact_avg > 0 else "N/A")

    # Fuzzy matching benchmarks
    print("\nFUZZY MATCHING:")
    print("-" * 40)

    prefixtrie_fuzzy_times = []
    rapidfuzz_fuzzy_times = []

    pt_fuzzy_results = None
    rf_fuzzy_results = None

    for i in range(num_runs):
        # PrefixTrie fuzzy
        results, time_taken = time_function(benchmark_prefixtrie_fuzzy, entries, queries, 2)
        prefixtrie_fuzzy_times.append(time_taken)
        if pt_fuzzy_results is None:
            pt_fuzzy_results = results

        # RapidFuzz fuzzy
        results, time_taken = time_function(benchmark_rapidfuzz_fuzzy, entries, queries, 80)
        rapidfuzz_fuzzy_times.append(time_taken)
        if rf_fuzzy_results is None:
            rf_fuzzy_results = results

    # Validate consistency
    validate_trie_consistency(entries, pt_fuzzy_results, f"{name} - PrefixTrie Fuzzy")
    validate_trie_consistency(entries, rf_fuzzy_results, f"{name} - RapidFuzz Fuzzy")

    pt_fuzzy_avg = statistics.mean(prefixtrie_fuzzy_times)
    pt_fuzzy_std = statistics.stdev(prefixtrie_fuzzy_times) if len(prefixtrie_fuzzy_times) > 1 else 0
    rf_fuzzy_avg = statistics.mean(rapidfuzz_fuzzy_times)
    rf_fuzzy_std = statistics.stdev(rapidfuzz_fuzzy_times) if len(rapidfuzz_fuzzy_times) > 1 else 0

    print(f"PrefixTrie:  {pt_fuzzy_avg:.4f}s ± {pt_fuzzy_std:.4f}s")
    print(f"RapidFuzz:   {rf_fuzzy_avg:.4f}s ± {rf_fuzzy_std:.4f}s")
    print(f"Speedup:     {rf_fuzzy_avg/pt_fuzzy_avg:.2f}x" if pt_fuzzy_avg > 0 else "N/A")

    return {
        'name': name,
        'entries_count': len(entries),
        'queries_count': len(queries),
        'exact': {
            'prefixtrie': {'avg': pt_exact_avg, 'std': pt_exact_std},
            'rapidfuzz': {'avg': rf_exact_avg, 'std': rf_exact_std},
            'speedup': rf_exact_avg/pt_exact_avg if pt_exact_avg > 0 else float('inf')
        },
        'fuzzy': {
            'prefixtrie': {'avg': pt_fuzzy_avg, 'std': pt_fuzzy_std},
            'rapidfuzz': {'avg': rf_fuzzy_avg, 'std': rf_fuzzy_std},
            'speedup': rf_fuzzy_avg/pt_fuzzy_avg if pt_fuzzy_avg > 0 else float('inf')
        }
    }


class TestBenchmarks:
    """Benchmark tests comparing PrefixTrie vs RapidFuzz"""

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_small_dataset_benchmark(self):
        """Benchmark with small dataset (100 entries)"""
        entries = generate_random_strings(100, 10)
        queries = generate_queries_with_errors(entries[:50])  # 50 queries

        result = run_benchmark_suite("Small Dataset", entries, queries, num_runs=3)

        # Basic assertions - just ensure benchmarks complete
        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_medium_dataset_benchmark(self):
        """Benchmark with medium dataset (1000 entries)"""
        entries = generate_random_strings(1000, 12)
        queries = generate_queries_with_errors(entries[:200])  # 200 queries

        result = run_benchmark_suite("Medium Dataset", entries, queries, num_runs=3)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_large_dataset_benchmark(self):
        """Benchmark with large dataset (10000 entries)"""
        entries = generate_random_strings(10000, 15)
        queries = generate_queries_with_errors(entries[:1000])  # 1000 queries

        result = run_benchmark_suite("Large Dataset", entries, queries, num_runs=2)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_very_large_dataset_benchmark(self):
        """Benchmark with very large dataset (50000 entries)"""
        entries = generate_random_strings(50000, 20)
        queries = generate_queries_with_errors(entries[:2000])  # 2000 queries

        result = run_benchmark_suite("Very Large Dataset", entries, queries, num_runs=2)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_massive_dataset_benchmark(self):
        """Benchmark with massive dataset (100000 entries)"""
        entries = generate_random_strings(100000, 25)
        queries = generate_queries_with_errors(entries[:3000])  # 3000 queries

        result = run_benchmark_suite("Massive Dataset", entries, queries, num_runs=1)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_dna_sequences_benchmark(self):
        """Benchmark with DNA sequences (bioinformatics use case)"""
        entries = generate_dna_sequences(10000, 50)  # 10k DNA sequences, 50bp each
        queries = generate_queries_with_errors(entries[:1000])  # 1000 queries

        result = run_benchmark_suite("DNA Sequences", entries, queries, num_runs=3)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_long_dna_sequences_benchmark(self):
        """Benchmark with long DNA sequences"""
        entries = generate_dna_sequences(5000, 200)  # 5k sequences, 200bp each
        queries = generate_queries_with_errors(entries[:500])  # 500 queries

        result = run_benchmark_suite("Long DNA Sequences", entries, queries, num_runs=2)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_protein_sequences_benchmark(self):
        """Benchmark with protein sequences"""
        entries = generate_protein_sequences(8000, 100)  # 8k protein sequences, 100aa each
        queries = generate_queries_with_errors(entries[:800])  # 800 queries

        result = run_benchmark_suite("Protein Sequences", entries, queries, num_runs=2)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_short_strings_benchmark(self):
        """Benchmark with very short strings"""
        entries = generate_random_strings(20000, 4)  # Many very short strings
        queries = generate_queries_with_errors(entries[:2000])

        result = run_benchmark_suite("Short Strings", entries, queries, num_runs=3)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_long_strings_benchmark(self):
        """Benchmark with very long strings"""
        entries = generate_random_strings(2000, 200)  # Long strings
        queries = generate_queries_with_errors(entries[:200])

        result = run_benchmark_suite("Long Strings", entries, queries, num_runs=3)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_very_long_strings_benchmark(self):
        """Benchmark with extremely long strings"""
        entries = generate_random_strings(500, 1000)  # Very long strings
        queries = generate_queries_with_errors(entries[:100])

        result = run_benchmark_suite("Very Long Strings", entries, queries, num_runs=2)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_high_similarity_benchmark(self):
        """Benchmark with highly similar strings (worst case for tries)"""
        # Create strings with common prefixes
        base_strings = ["prefix_" + str(i).zfill(4) for i in range(5000)]
        entries = base_strings + [s + "_suffix" for s in base_strings[:2500]]
        queries = generate_queries_with_errors(entries[:1000])

        result = run_benchmark_suite("High Similarity", entries, queries, num_runs=3)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_realistic_words_benchmark(self):
        """Benchmark with realistic English-like words"""
        entries = generate_realistic_words(15000)
        queries = generate_queries_with_errors(entries[:1500])

        result = run_benchmark_suite("Realistic Words", entries, queries, num_runs=3)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_hierarchical_strings_benchmark(self):
        """Benchmark with hierarchical/path-like strings"""
        entries = generate_hierarchical_strings(12000, 4)
        queries = generate_queries_with_errors(entries[:1200])

        result = run_benchmark_suite("Hierarchical Strings", entries, queries, num_runs=3)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0

    @pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz not available")
    def test_mixed_length_benchmark(self):
        """Benchmark with mixed string lengths"""
        entries = []
        entries.extend(generate_random_strings(5000, 5))   # Short
        entries.extend(generate_random_strings(3000, 20))  # Medium
        entries.extend(generate_random_strings(1500, 50))  # Long
        entries.extend(generate_random_strings(500, 150))  # Very long
        random.shuffle(entries)

        queries = generate_queries_with_errors(entries[:1500])

        result = run_benchmark_suite("Mixed Lengths", entries, queries, num_runs=3)

        assert result['exact']['prefixtrie']['avg'] > 0
        assert result['exact']['rapidfuzz']['avg'] > 0
        assert result['fuzzy']['prefixtrie']['avg'] > 0
        assert result['fuzzy']['rapidfuzz']['avg'] > 0


def run_full_benchmark_suite():
    """Run the complete benchmark suite and print summary"""
    print("\n" + "="*80)
    print("FULL BENCHMARK SUITE - PrefixTrie vs RapidFuzz")
    print("="*80)

    # Set random seed for reproducible results
    random.seed(42)

    all_results = []

    # Enhanced dataset configurations with much larger sizes
    benchmark_configs = [
        ("Small Random", lambda: (generate_random_strings(500, 8), 100)),
        ("Medium Random", lambda: (generate_random_strings(5000, 12), 500)),
        ("Large Random", lambda: (generate_random_strings(25000, 15), 1500)),
        ("Very Large Random", lambda: (generate_random_strings(75000, 20), 3000)),
        ("Massive Random", lambda: (generate_random_strings(150000, 25), 5000)),
        ("DNA Sequences", lambda: (generate_dna_sequences(15000, 50), 1000)),
        ("Long DNA", lambda: (generate_dna_sequences(8000, 150), 800)),
        ("Protein Sequences", lambda: (generate_protein_sequences(10000, 80), 1000)),
        ("Short Strings", lambda: (generate_random_strings(30000, 4), 2000)),
        ("Long Strings", lambda: (generate_random_strings(3000, 200), 300)),
        ("Very Long Strings", lambda: (generate_random_strings(1000, 500), 150)),
        ("Realistic Words", lambda: (generate_realistic_words(20000), 1500)),
        ("Hierarchical", lambda: (generate_hierarchical_strings(15000, 4), 1200)),
        ("Common Prefixes", lambda: (["prefix_" + str(i).zfill(4) for i in range(20000)], 1500)),
        ("Mixed Lengths", lambda: create_mixed_length_dataset()),
    ]

    for name, config_func in benchmark_configs:
        try:
            entries, query_count = config_func()
            queries = generate_queries_with_errors(entries[:query_count])

            result = run_benchmark_suite(name, entries, queries, num_runs=2)
            all_results.append(result)
        except Exception as e:
            print(f"Error in benchmark '{name}': {e}")
            continue

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Benchmark':<20} {'Entries':<8} {'Queries':<8} {'Exact Speedup':<13} {'Fuzzy Speedup':<13}")
    print("-" * 80)

    for result in all_results:
        exact_speedup = result['exact']['speedup']
        fuzzy_speedup = result['fuzzy']['speedup']

        exact_str = f"{exact_speedup:.2f}x" if exact_speedup != float('inf') else "∞"
        fuzzy_str = f"{fuzzy_speedup:.2f}x" if fuzzy_speedup != float('inf') else "∞"

        print(f"{result['name']:<20} {result['entries_count']:<8} {result['queries_count']:<8} "
              f"{exact_str:<13} {fuzzy_str:<13}")

    # Calculate averages
    exact_speedups = [r['exact']['speedup'] for r in all_results if r['exact']['speedup'] != float('inf')]
    fuzzy_speedups = [r['fuzzy']['speedup'] for r in all_results if r['fuzzy']['speedup'] != float('inf')]

    if exact_speedups:
        avg_exact = statistics.mean(exact_speedups)
        print(f"\nAverage exact matching speedup: {avg_exact:.2f}x")

    if fuzzy_speedups:
        avg_fuzzy = statistics.mean(fuzzy_speedups)
        print(f"Average fuzzy matching speedup: {avg_fuzzy:.2f}x")

    return all_results


def create_mixed_length_dataset():
    """Create a mixed dataset with various string lengths"""
    entries = []
    entries.extend(generate_random_strings(8000, 5))   # Short
    entries.extend(generate_random_strings(5000, 15))  # Medium
    entries.extend(generate_random_strings(3000, 40))  # Long
    entries.extend(generate_random_strings(1500, 100)) # Very long
    entries.extend(generate_random_strings(500, 300))  # Extremely long
    random.shuffle(entries)
    return entries, 2000


if __name__ == "__main__":
    # Run the full benchmark suite when script is executed directly
    if RAPIDFUZZ_AVAILABLE:
        run_full_benchmark_suite()
    else:
        print("RapidFuzz not available. Install with: pip install rapidfuzz")
