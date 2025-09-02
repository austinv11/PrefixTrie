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
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path so we can import prefixtrie
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pyximport

    pyximport.install(
        setup_args={"include_dirs": ["../src/prefixtrie"]},
    )
    from prefixtrie import PrefixTrie
    print("✓ PrefixTrie imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PrefixTrie: {e}")
    sys.exit(1)

try:
    import rapidfuzz
    from rapidfuzz import process
    print(f"✓ RapidFuzz imported successfully (version {rapidfuzz.__version__})")
    RAPIDFUZZ_AVAILABLE = True
except ImportError as e:
    print(f"✗ RapidFuzz not available: {e}")
    print("Install with: pip install rapidfuzz")
    RAPIDFUZZ_AVAILABLE = False

try:
    from thefuzz import process as fuzz_process
    print("✓ TheFuzz imported successfully")
    THEFUZZ_AVAILABLE = True
except ImportError as e:
    print(f"✗ TheFuzz not available: {e}")
    print("Install with: pip install thefuzz[speedup]")
    THEFUZZ_AVAILABLE = False

try:
    from symspellpy import SymSpell, Verbosity
    print("✓ SymSpellPy imported successfully")
    SYMSPELL_AVAILABLE = True
except ImportError as e:
    print(f"✗ SymSpellPy not available: {e}")
    print("Install with: pip install symspellpy")
    SYMSPELL_AVAILABLE = False


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


def benchmark_thefuzz(entries, queries, score_cutoff=80):
    """Benchmark TheFuzz performance"""
    if not THEFUZZ_AVAILABLE:
        return None, 0, 0

    print(f"Running TheFuzz on {len(entries)} entries with {len(queries)} queries...")
    build_time = 0  # No build phase

    start_search = time.perf_counter()
    results = []
    for query in queries:
        match = fuzz_process.extractOne(query, entries, score_cutoff=score_cutoff)
        if match:
            results.append((match[0], match[1] == 100))
        else:
            results.append((None, False))
    search_time = time.perf_counter() - start_search

    return results, build_time, search_time


def benchmark_symspell(entries, queries, max_edit_distance=2):
    """Benchmark SymSpellPy performance"""
    if not SYMSPELL_AVAILABLE:
        return None, 0, 0

    print(f"Building SymSpell dictionary with {len(entries)} entries...")
    start_build = time.perf_counter()
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
    for entry in entries:
        sym_spell.create_dictionary_entry(entry, 1)
    build_time = time.perf_counter() - start_build

    print(f"Running {len(queries)} SymSpell lookups...")
    start_search = time.perf_counter()
    results = []
    for query in queries:
        suggestions = sym_spell.lookup(query, Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
        if suggestions:
            best_suggestion = suggestions[0]
            results.append((best_suggestion.term, best_suggestion.distance == 0))
        else:
            results.append((None, False))
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
    rf_build_times, rf_search_times, rf_total_times, rf_results = [], [], [], None
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

    # Benchmark TheFuzz multiple times
    tf_build_times, tf_search_times, tf_total_times, tf_results = [], [], [], None
    if THEFUZZ_AVAILABLE:
        print(f"\nRunning TheFuzz benchmark ({num_runs} runs)...")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            results, build_time, search_time = benchmark_thefuzz(entries, queries)
            tf_build_times.append(build_time)
            tf_search_times.append(search_time)
            tf_total_times.append(build_time + search_time)
            if tf_results is None:
                tf_results = results

    # Benchmark SymSpell multiple times
    ss_build_times, ss_search_times, ss_total_times, ss_results = [], [], [], None
    if SYMSPELL_AVAILABLE:
        print(f"\nRunning SymSpell benchmark ({num_runs} runs)...")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            results, build_time, search_time = benchmark_symspell(entries, queries)
            ss_build_times.append(build_time)
            ss_search_times.append(search_time)
            ss_total_times.append(build_time + search_time)
            if ss_results is None:
                ss_results = results

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
        rf_build_avg, _ = calc_stats(rf_build_times)
        rf_search_avg, _ = calc_stats(rf_search_times)
        rf_total_avg, _ = calc_stats(rf_total_times)
    else:
        rf_build_avg, rf_search_avg, rf_total_avg = 0, 0, 0

    if THEFUZZ_AVAILABLE:
        tf_build_avg, _ = calc_stats(tf_build_times)
        tf_search_avg, _ = calc_stats(tf_search_times)
        tf_total_avg, _ = calc_stats(tf_total_times)
    else:
        tf_build_avg, tf_search_avg, tf_total_avg = 0, 0, 0

    if SYMSPELL_AVAILABLE:
        ss_build_avg, _ = calc_stats(ss_build_times)
        ss_search_avg, _ = calc_stats(ss_search_times)
        ss_total_avg, _ = calc_stats(ss_total_times)
    else:
        ss_build_avg, ss_search_avg, ss_total_avg = 0, 0, 0

    # Print results
    print(f"\n{'Implementation':<20} {'Build Time (s)':<20} {'Search Time (s)':<20} {'Total Time (s)':<20}")
    print("-" * 80)
    print(f"{'PrefixTrie':<20} {pt_build_avg:<20.4f} {pt_search_avg:<20.4f} {pt_total_avg:<20.4f}")
    if RAPIDFUZZ_AVAILABLE:
        print(f"{'RapidFuzz':<20} {rf_build_avg:<20.4f} {rf_search_avg:<20.4f} {rf_total_avg:<20.4f}")
    if THEFUZZ_AVAILABLE:
        print(f"{'TheFuzz':<20} {tf_build_avg:<20.4f} {tf_search_avg:<20.4f} {tf_total_avg:<20.4f}")
    if SYMSPELL_AVAILABLE:
        print(f"{'SymSpell':<20} {ss_build_avg:<20.4f} {ss_search_avg:<20.4f} {ss_total_avg:<20.4f}")

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
            'build_avg': rf_build_avg,
            'search_avg': rf_search_avg,
            'total_avg': rf_total_avg,
        } if RAPIDFUZZ_AVAILABLE else None,
        'thefuzz': {
            'build_avg': tf_build_avg,
            'search_avg': tf_search_avg,
            'total_avg': tf_total_avg,
        } if THEFUZZ_AVAILABLE else None,
        'symspell': {
            'build_avg': ss_build_avg,
            'search_avg': ss_search_avg,
            'total_avg': ss_total_avg,
        } if SYMSPELL_AVAILABLE else None
    }


def main():
    """Run the benchmark suite"""
    import argparse
    parser = argparse.ArgumentParser(description="PrefixTrie vs Competitors Benchmark Suite")
    parser.add_argument("--output-plot", default="benchmark_search.png", help="Output file for the benchmark plot")
    args = parser.parse_args()

    print("PrefixTrie vs Competitors Benchmark Suite")
    print("=" * 60)

    if not RAPIDFUZZ_AVAILABLE:
        print("Warning: RapidFuzz not available. Only testing PrefixTrie.")
    if not THEFUZZ_AVAILABLE:
        print("Warning: TheFuzz not available. Only testing PrefixTrie.")
    if not SYMSPELL_AVAILABLE:
        print("Warning: SymSpellPy not available. Only testing PrefixTrie.")

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

    # Generate plot
    scenarios = [r['scenario'] for r in all_results]
    pt_search_times = [r['prefixtrie']['search_avg'] for r in all_results]
    rf_search_times = [r['rapidfuzz']['search_avg'] if r.get('rapidfuzz') else 0 for r in all_results]
    tf_search_times = [r['thefuzz']['search_avg'] if r.get('thefuzz') else 0 for r in all_results]
    ss_search_times = [r['symspell']['search_avg'] if r.get('symspell') else 0 for r in all_results]

    x = np.arange(len(scenarios))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(x - width*1.5, pt_search_times, width, label='PrefixTrie')
    if RAPIDFUZZ_AVAILABLE:
        ax.bar(x - width*0.5, rf_search_times, width, label='RapidFuzz')
    if THEFUZZ_AVAILABLE:
        ax.bar(x + width*0.5, tf_search_times, width, label='TheFuzz')
    if SYMSPELL_AVAILABLE:
        ax.bar(x + width*1.5, ss_search_times, width, label='SymSpell')

    ax.set_ylabel('Search Time (s) (Lower is better)')
    ax.set_title('Benchmark: Search Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Scenario')
    fig.tight_layout()
    plt.savefig(args.output_plot)

    print(f"\nBenchmark plot saved to {args.output_plot}")


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
    rf_build_times, rf_search_times, rf_total_times, rf_results = [], [], [], None
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

    # Benchmark TheFuzz multiple times
    tf_build_times, tf_search_times, tf_total_times, tf_results = [], [], [], None
    if THEFUZZ_AVAILABLE:
        print(f"\nRunning TheFuzz benchmark ({num_runs} runs)...")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            results, build_time, search_time = benchmark_thefuzz(entries, queries)
            tf_build_times.append(build_time)
            tf_search_times.append(search_time)
            tf_total_times.append(build_time + search_time)
            if tf_results is None:
                tf_results = results

    # Benchmark SymSpell multiple times
    ss_build_times, ss_search_times, ss_total_times, ss_results = [], [], [], None
    if SYMSPELL_AVAILABLE:
        print(f"\nRunning SymSpell benchmark ({num_runs} runs)...")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            results, build_time, search_time = benchmark_symspell(entries, queries)
            ss_build_times.append(build_time)
            ss_search_times.append(search_time)
            ss_total_times.append(build_time + search_time)
            if ss_results is None:
                ss_results = results

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
        rf_build_avg, _ = calc_stats(rf_build_times)
        rf_search_avg, _ = calc_stats(rf_search_times)
        rf_total_avg, _ = calc_stats(rf_total_times)
    else:
        rf_build_avg, rf_search_avg, rf_total_avg = 0, 0, 0

    if THEFUZZ_AVAILABLE:
        tf_build_avg, _ = calc_stats(tf_build_times)
        tf_search_avg, _ = calc_stats(tf_search_times)
        tf_total_avg, _ = calc_stats(tf_total_times)
    else:
        tf_build_avg, tf_search_avg, tf_total_avg = 0, 0, 0

    if SYMSPELL_AVAILABLE:
        ss_build_avg, _ = calc_stats(ss_build_times)
        ss_search_avg, _ = calc_stats(ss_search_times)
        ss_total_avg, _ = calc_stats(ss_total_times)
    else:
        ss_build_avg, ss_search_avg, ss_total_avg = 0, 0, 0

    # Print results
    print(f"\n{'Implementation':<20} {'Build Time (s)':<20} {'Search Time (s)':<20} {'Total Time (s)':<20}")
    print("-" * 80)
    print(f"{'PrefixTrie':<20} {pt_build_avg:<20.4f} {pt_search_avg:<20.4f} {pt_total_avg:<20.4f}")
    if RAPIDFUZZ_AVAILABLE:
        print(f"{'RapidFuzz':<20} {rf_build_avg:<20.4f} {rf_search_avg:<20.4f} {rf_total_avg:<20.4f}")
    if THEFUZZ_AVAILABLE:
        print(f"{'TheFuzz':<20} {tf_build_avg:<20.4f} {tf_search_avg:<20.4f} {tf_total_avg:<20.4f}")
    if SYMSPELL_AVAILABLE:
        print(f"{'SymSpell':<20} {ss_build_avg:<20.4f} {ss_search_avg:<20.4f} {ss_total_avg:<20.4f}")

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
            'build_avg': rf_build_avg,
            'search_avg': rf_search_avg,
            'total_avg': rf_total_avg,
        } if RAPIDFUZZ_AVAILABLE else None,
        'thefuzz': {
            'build_avg': tf_build_avg,
            'search_avg': tf_search_avg,
            'total_avg': tf_total_avg,
        } if THEFUZZ_AVAILABLE else None,
        'symspell': {
            'build_avg': ss_build_avg,
            'search_avg': ss_search_avg,
            'total_avg': ss_total_avg,
        } if SYMSPELL_AVAILABLE else None,
        'entries_count': len(entries),
        'queries_count': len(queries)
    }


if __name__ == "__main__":
    main()
