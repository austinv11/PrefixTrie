#!/usr/bin/env python3
"""
Simple benchmark script to compare PrefixTrie vs RapidFuzz performance.
Run this script directly to see benchmark results.
"""

import time
import statistics
import random
import sys
import os

# Add the src directory to the path so we can import prefixtrie
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pyximport
    pyximport.install()
    from prefixtrie import PrefixTrie
    print("✓ PrefixTrie imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PrefixTrie: {e}")
    sys.exit(1)

try:
    import rapidfuzz
    from rapidfuzz import process, fuzz
    print(f"✓ RapidFuzz imported successfully (version {rapidfuzz.__version__})")
    RAPIDFUZZ_AVAILABLE = True
except ImportError as e:
    print(f"✗ RapidFuzz not available: {e}")
    print("Install with: pip install rapidfuzz")
    RAPIDFUZZ_AVAILABLE = False


def generate_random_strings(n: int, length: int = 10, alphabet: str = None) -> list[str]:
    """Generate n random strings of given length"""
    if alphabet is None:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'

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


def validate_trie_consistency(entries: list[str], trie_results: list[tuple], test_name: str = ""):
    """Validate that trie results are consistent with expected behavior"""
    print(f"  Validating consistency for {test_name}...")

    entries_set = set(entries)
    inconsistencies = []

    for i, (result, corrections) in enumerate(trie_results):
        if result is not None:
            # If result is found, it should be in the original entries
            if result not in entries_set:
                inconsistencies.append(f"Index {i}: Found '{result}' not in original entries")

    if inconsistencies:
        print(f"  WARNING: Found {len(inconsistencies)} inconsistencies:")
        for inc in inconsistencies[:3]:  # Show first 3
            print(f"    {inc}")
        if len(inconsistencies) > 3:
            print(f"    ... and {len(inconsistencies) - 3} more")
    else:
        print(f"  ✓ No inconsistencies found")

    return len(inconsistencies) == 0


def generate_test_data(n_entries=1000, n_queries=200, string_length=12):
    """Generate test data for benchmarking"""
    print(f"Generating {n_entries} entries and {n_queries} queries...")

    # Generate random entries
    entries = []
    for i in range(n_entries):
        # Mix of random strings and structured strings
        if i % 3 == 0:
            # DNA-like sequences
            entry = ''.join(random.choices('ATCG', k=string_length))
        elif i % 3 == 1:
            # Random lowercase strings
            entry = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=string_length))
        else:
            # Structured strings
            entry = f"item_{i:06d}_{random.randint(1000, 9999)}"
        entries.append(entry)

    # Generate queries (mix of exact matches and fuzzy variants)
    queries = []

    # Add some exact matches
    for i in range(n_queries // 2):
        queries.append(random.choice(entries))

    # Add some fuzzy variants
    for i in range(n_queries // 2):
        base = random.choice(entries)
        if len(base) > 2:
            # Create a variant with 1-2 character changes
            variant = list(base)
            for _ in range(random.randint(1, 2)):
                if variant:
                    pos = random.randint(0, len(variant) - 1)
                    variant[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
            queries.append(''.join(variant))
        else:
            queries.append(base)

    return entries, queries


def benchmark_prefixtrie(entries, queries, allow_indels=True, correction_budget=2):
    """Benchmark PrefixTrie performance"""
    print(f"Building PrefixTrie with {len(entries)} entries...")
    start_build = time.perf_counter()
    trie = PrefixTrie(entries, allow_indels=allow_indels)
    build_time = time.perf_counter() - start_build

    print(f"Running {len(queries)} searches...")
    start_search = time.perf_counter()
    results = []
    for query in queries:
        result, corrections = trie.search(query, correction_budget=correction_budget)
        results.append((result, corrections))
    search_time = time.perf_counter() - start_search

    return results, build_time, search_time


def benchmark_rapidfuzz(entries, queries, score_cutoff=80):
    """Benchmark RapidFuzz performance"""
    if not RAPIDFUZZ_AVAILABLE:
        return None, 0, 0

    print(f"Running RapidFuzz on {len(entries)} entries with {len(queries)} queries...")

    # RapidFuzz doesn't have a "build" phase like tries, so build_time is 0
    build_time = 0

    start_search = time.perf_counter()
    results = []
    for query in queries:
        match = process.extractOne(query, entries, score_cutoff=score_cutoff)
        if match:
            results.append((match[0], match[1] == 100))  # exact if score is 100
        else:
            results.append((None, False))
    search_time = time.perf_counter() - start_search

    return results, build_time, search_time


def run_benchmark(n_entries=1000, n_queries=200, string_length=12, num_runs=3):
    """Run a complete benchmark comparison"""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {n_entries} entries, {n_queries} queries, {string_length} chars")
    print(f"{'='*60}")

    # Generate test data once
    entries, queries = generate_test_data(n_entries, n_queries, string_length)

    # Benchmark PrefixTrie multiple times
    pt_build_times = []
    pt_search_times = []
    pt_total_times = []
    pt_results = None

    print(f"\nRunning PrefixTrie benchmark ({num_runs} runs)...")
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        results, build_time, search_time = benchmark_prefixtrie(entries, queries)
        pt_build_times.append(build_time)
        pt_search_times.append(search_time)
        pt_total_times.append(build_time + search_time)
        if pt_results is None:
            pt_results = results

    # Benchmark RapidFuzz multiple times
    rf_build_times = []
    rf_search_times = []
    rf_total_times = []
    rf_results = None

    if RAPIDFUZZ_AVAILABLE:
        print(f"\nRunning RapidFuzz benchmark ({num_runs} runs)...")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            results, build_time, search_time = benchmark_rapidfuzz(entries, queries)
            rf_build_times.append(build_time)
            rf_search_times.append(search_time)
            rf_total_times.append(build_time + search_time)
            if rf_results is None:
                rf_results = results

    # Calculate statistics
    def calc_stats(times):
        if not times:
            return 0, 0
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        return avg, std

    pt_build_avg, pt_build_std = calc_stats(pt_build_times)
    pt_search_avg, pt_search_std = calc_stats(pt_search_times)
    pt_total_avg, pt_total_std = calc_stats(pt_total_times)

    # Print results
    print(f"\n{'Results':<20} {'Avg Time':<12} {'Std Dev':<12}")
    print("-" * 50)
    print(f"{'PrefixTrie Build':<20} {pt_build_avg:.4f}s{'':<4} {pt_build_std:.4f}s")
    print(f"{'PrefixTrie Search':<20} {pt_search_avg:.4f}s{'':<4} {pt_search_std:.4f}s")
    print(f"{'PrefixTrie Total':<20} {pt_total_avg:.4f}s{'':<4} {pt_total_std:.4f}s")

    if RAPIDFUZZ_AVAILABLE:
        rf_build_avg, rf_build_std = calc_stats(rf_build_times)
        rf_search_avg, rf_search_std = calc_stats(rf_search_times)
        rf_total_avg, rf_total_std = calc_stats(rf_total_times)

        print(f"{'RapidFuzz Build':<20} {rf_build_avg:.4f}s{'':<4} {rf_build_std:.4f}s")
        print(f"{'RapidFuzz Search':<20} {rf_search_avg:.4f}s{'':<4} {rf_search_std:.4f}s")
        print(f"{'RapidFuzz Total':<20} {rf_total_avg:.4f}s{'':<4} {rf_total_std:.4f}s")

        # Calculate speedups
        if pt_search_avg > 0 and rf_search_avg > 0:
            search_speedup = rf_search_avg / pt_search_avg
            total_speedup = rf_total_avg / pt_total_avg

            print(f"\nSpeedup Analysis:")
            print(f"Search speedup: {search_speedup:.2f}x {'(PrefixTrie faster)' if search_speedup > 1 else '(RapidFuzz faster)'}")
            print(f"Total speedup:  {total_speedup:.2f}x {'(PrefixTrie faster)' if total_speedup > 1 else '(RapidFuzz faster)'}")

    # Analyze result quality
    if pt_results and rf_results and RAPIDFUZZ_AVAILABLE:
        pt_found = sum(1 for r, _ in pt_results if r is not None)
        rf_found = sum(1 for r, _ in rf_results if r is not None)

        print(f"\nResult Quality:")
        print(f"PrefixTrie found: {pt_found}/{len(queries)} queries ({pt_found/len(queries)*100:.1f}%)")
        print(f"RapidFuzz found:  {rf_found}/{len(queries)} queries ({rf_found/len(queries)*100:.1f}%)")

    # Validate consistency of results
    print(f"\nValidating consistency of results...")
    pt_consistent = validate_trie_consistency(entries, pt_results, "PrefixTrie")
    rf_consistent = True  # RapidFuzz consistency is not applicable in the same way

    return {
        'prefixtrie': {
            'build_avg': pt_build_avg,
            'search_avg': pt_search_avg,
            'total_avg': pt_total_avg,
            'consistent': pt_consistent
        },
        'rapidfuzz': {
            'build_avg': rf_build_avg if RAPIDFUZZ_AVAILABLE else 0,
            'search_avg': rf_search_avg if RAPIDFUZZ_AVAILABLE else 0,
            'total_avg': rf_total_avg if RAPIDFUZZ_AVAILABLE else 0
        } if RAPIDFUZZ_AVAILABLE else None
    }


def main():
    """Run the benchmark suite"""
    print("PrefixTrie vs RapidFuzz Benchmark Suite")
    print("=" * 60)

    if not RAPIDFUZZ_AVAILABLE:
        print("Warning: RapidFuzz not available. Only testing PrefixTrie.")
        print("Install RapidFuzz with: pip install rapidfuzz")

    # Set random seed for reproducible results
    random.seed(42)

    # Enhanced benchmark scenarios with much larger datasets
    scenarios = [
        # Standard size progression
        {"name": "Small Dataset", "n_entries": 500, "n_queries": 100, "string_length": 8},
        {"name": "Medium Dataset", "n_entries": 5000, "n_queries": 500, "string_length": 12},
        {"name": "Large Dataset", "n_entries": 25000, "n_queries": 1500, "string_length": 15},
        {"name": "Very Large Dataset", "n_entries": 75000, "n_queries": 3000, "string_length": 20},
        {"name": "Massive Dataset", "n_entries": 150000, "n_queries": 5000, "string_length": 25},

        # String length variations
        {"name": "Very Short Strings", "n_entries": 30000, "n_queries": 2000, "string_length": 3},
        {"name": "Short Strings", "n_entries": 20000, "n_queries": 1500, "string_length": 6},
        {"name": "Long Strings", "n_entries": 3000, "n_queries": 300, "string_length": 100},
        {"name": "Very Long Strings", "n_entries": 1000, "n_queries": 150, "string_length": 300},
        {"name": "Extremely Long", "n_entries": 500, "n_queries": 75, "string_length": 1000},
    ]

    all_results = []

    for scenario in scenarios:
        print(f"\n\nRunning scenario: {scenario['name']}")
        try:
            result = run_benchmark(
                n_entries=scenario['n_entries'],
                n_queries=scenario['n_queries'],
                string_length=scenario['string_length'],
                num_runs=2  # Reduced runs for larger datasets
            )
            result['scenario'] = scenario['name']
            all_results.append(result)
        except Exception as e:
            print(f"Error in scenario '{scenario['name']}': {e}")
            continue

    # Add specialized data type benchmarks
    print("\n\nRunning specialized data type benchmarks...")

    specialized_scenarios = [
        {
            "name": "DNA Sequences",
            "generator": lambda: generate_dna_sequences(15000, 50),
            "queries": 1000
        },
        {
            "name": "Long DNA Sequences",
            "generator": lambda: generate_dna_sequences(8000, 200),
            "queries": 800
        },
        {
            "name": "Protein Sequences",
            "generator": lambda: generate_protein_sequences(10000, 100),
            "queries": 1000
        },
        {
            "name": "Realistic Words",
            "generator": lambda: generate_realistic_words(20000),
            "queries": 1500
        },
        {
            "name": "Hierarchical Paths",
            "generator": lambda: generate_hierarchical_strings(15000, 4),
            "queries": 1200
        },
        {
            "name": "Common Prefixes",
            "generator": lambda: [f"prefix_{i:06d}_suffix" for i in range(25000)],
            "queries": 2000
        }
    ]

    for spec in specialized_scenarios:
        print(f"\n\nRunning specialized benchmark: {spec['name']}")
        try:
            entries = spec['generator']()
            queries = []

            # Generate queries with errors
            for _ in range(spec['queries']):
                if random.random() < 0.5:
                    # Exact match
                    queries.append(random.choice(entries))
                else:
                    # Create variant with errors
                    base = random.choice(entries)
                    if len(base) > 2:
                        variant = list(base)
                        for _ in range(random.randint(1, 2)):
                            if variant:
                                pos = random.randint(0, len(variant) - 1)
                                # Use appropriate alphabet for mutations
                                if 'DNA' in spec['name']:
                                    variant[pos] = random.choice('ATCG')
                                elif 'Protein' in spec['name']:
                                    variant[pos] = random.choice('ACDEFGHIKLMNPQRSTVWY')
                                else:
                                    variant[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
                        queries.append(''.join(variant))
                    else:
                        queries.append(base)

            result = run_specialized_benchmark(spec['name'], entries, queries, num_runs=2)
            result['scenario'] = spec['name']
            all_results.append(result)
        except Exception as e:
            print(f"Error in specialized scenario '{spec['name']}': {e}")
            continue

    # Print comprehensive summary
    print(f"\n\n{'='*100}")
    print("COMPREHENSIVE SUMMARY")
    print("="*100)
    print(f"{'Scenario':<25} {'Entries':<8} {'Queries':<8} {'PT Search':<10} {'RF Search':<10} {'PT Total':<10} {'RF Total':<10} {'Speedup':<8}")
    print("-" * 100)

    for result in all_results:
        pt_search = result['prefixtrie']['search_avg']
        pt_total = result['prefixtrie']['total_avg']
        entries_count = result.get('entries_count', 'N/A')
        queries_count = result.get('queries_count', 'N/A')

        if result['rapidfuzz']:
            rf_search = result['rapidfuzz']['search_avg']
            rf_total = result['rapidfuzz']['total_avg']
            speedup = rf_search / pt_search if pt_search > 0 else 0

            print(f"{result['scenario']:<25} {entries_count:<8} {queries_count:<8} "
                  f"{pt_search:.4f}s{'':<2} {rf_search:.4f}s{'':<2} "
                  f"{pt_total:.4f}s{'':<2} {rf_total:.4f}s{'':<2} {speedup:.2f}x")
        else:
            print(f"{result['scenario']:<25} {entries_count:<8} {queries_count:<8} "
                  f"{pt_search:.4f}s{'':<2} {'N/A':<10} "
                  f"{pt_total:.4f}s{'':<2} {'N/A':<10} {'N/A':<8}")

    # Calculate overall statistics
    if RAPIDFUZZ_AVAILABLE:
        search_speedups = []
        total_speedups = []

        for result in all_results:
            if result['rapidfuzz'] and result['prefixtrie']['search_avg'] > 0:
                search_speedup = result['rapidfuzz']['search_avg'] / result['prefixtrie']['search_avg']
                total_speedup = result['rapidfuzz']['total_avg'] / result['prefixtrie']['total_avg']

                search_speedups.append(search_speedup)
                total_speedups.append(total_speedup)

        if search_speedups:
            avg_search_speedup = statistics.mean(search_speedups)
            avg_total_speedup = statistics.mean(total_speedups)
            median_search_speedup = statistics.median(search_speedups)
            median_total_speedup = statistics.median(total_speedups)

            print(f"\nOVERALL PERFORMANCE ANALYSIS:")
            print(f"Average search speedup: {avg_search_speedup:.2f}x {'(PrefixTrie faster)' if avg_search_speedup > 1 else '(RapidFuzz faster)'}")
            print(f"Median search speedup:  {median_search_speedup:.2f}x {'(PrefixTrie faster)' if median_search_speedup > 1 else '(RapidFuzz faster)'}")
            print(f"Average total speedup:  {avg_total_speedup:.2f}x {'(PrefixTrie faster)' if avg_total_speedup > 1 else '(RapidFuzz faster)'}")
            print(f"Median total speedup:   {median_total_speedup:.2f}x {'(PrefixTrie faster)' if median_total_speedup > 1 else '(RapidFuzz faster)'}")

    print("\nBenchmark complete!")


def run_specialized_benchmark(name: str, entries: list[str], queries: list[str], num_runs: int = 3):
    """Run benchmark for specialized data types"""
    print(f"\n{'='*60}")
    print(f"SPECIALIZED BENCHMARK: {name}")
    print(f"Entries: {len(entries)}, Queries: {len(queries)}")
    print(f"{'='*60}")

    # Benchmark PrefixTrie multiple times
    pt_build_times = []
    pt_search_times = []
    pt_total_times = []
    pt_results = None

    print(f"\nRunning PrefixTrie benchmark ({num_runs} runs)...")
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        results, build_time, search_time = benchmark_prefixtrie(entries, queries)
        pt_build_times.append(build_time)
        pt_search_times.append(search_time)
        pt_total_times.append(build_time + search_time)
        if pt_results is None:
            pt_results = results

    # Benchmark RapidFuzz multiple times
    rf_build_times = []
    rf_search_times = []
    rf_total_times = []
    rf_results = None

    if RAPIDFUZZ_AVAILABLE:
        print(f"\nRunning RapidFuzz benchmark ({num_runs} runs)...")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            results, build_time, search_time = benchmark_rapidfuzz(entries, queries)
            rf_build_times.append(build_time)
            rf_search_times.append(search_time)
            rf_total_times.append(build_time + search_time)
            if rf_results is None:
                rf_results = results

    # Calculate statistics
    def calc_stats(times):
        if not times:
            return 0, 0
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        return avg, std

    pt_build_avg, pt_build_std = calc_stats(pt_build_times)
    pt_search_avg, pt_search_std = calc_stats(pt_search_times)
    pt_total_avg, pt_total_std = calc_stats(pt_total_times)

    # Print results
    print(f"\n{'Results':<20} {'Avg Time':<12} {'Std Dev':<12}")
    print("-" * 50)
    print(f"{'PrefixTrie Build':<20} {pt_build_avg:.4f}s{'':<4} {pt_build_std:.4f}s")
    print(f"{'PrefixTrie Search':<20} {pt_search_avg:.4f}s{'':<4} {pt_search_std:.4f}s")
    print(f"{'PrefixTrie Total':<20} {pt_total_avg:.4f}s{'':<4} {pt_total_std:.4f}s")

    if RAPIDFUZZ_AVAILABLE:
        rf_build_avg, rf_build_std = calc_stats(rf_build_times)
        rf_search_avg, rf_search_std = calc_stats(rf_search_times)
        rf_total_avg, rf_total_std = calc_stats(rf_total_times)

        print(f"{'RapidFuzz Build':<20} {rf_build_avg:.4f}s{'':<4} {rf_build_std:.4f}s")
        print(f"{'RapidFuzz Search':<20} {rf_search_avg:.4f}s{'':<4} {rf_search_std:.4f}s")
        print(f"{'RapidFuzz Total':<20} {rf_total_avg:.4f}s{'':<4} {rf_total_std:.4f}s")

        # Calculate speedups
        if pt_search_avg > 0 and rf_search_avg > 0:
            search_speedup = rf_search_avg / pt_search_avg
            total_speedup = rf_total_avg / pt_total_avg

            print(f"\nSpeedup Analysis:")
            print(f"Search speedup: {search_speedup:.2f}x {'(PrefixTrie faster)' if search_speedup > 1 else '(RapidFuzz faster)'}")
            print(f"Total speedup:  {total_speedup:.2f}x {'(PrefixTrie faster)' if total_speedup > 1 else '(RapidFuzz faster)'}")

    # Validate consistency of results
    print(f"\nValidating consistency of results...")
    pt_consistent = validate_trie_consistency(entries, pt_results, f"{name} - PrefixTrie")

    return {
        'prefixtrie': {
            'build_avg': pt_build_avg,
            'search_avg': pt_search_avg,
            'total_avg': pt_total_avg,
            'consistent': pt_consistent
        },
        'rapidfuzz': {
            'build_avg': rf_build_avg if RAPIDFUZZ_AVAILABLE else 0,
            'search_avg': rf_search_avg if RAPIDFUZZ_AVAILABLE else 0,
            'total_avg': rf_total_avg if RAPIDFUZZ_AVAILABLE else 0
        } if RAPIDFUZZ_AVAILABLE else None,
        'entries_count': len(entries),
        'queries_count': len(queries)
    }


if __name__ == "__main__":
    main()
