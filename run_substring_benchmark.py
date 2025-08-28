#!/usr/bin/env python3
"""
Benchmark script to compare PrefixTrie's substring search vs fuzzysearch performance.
Run this script directly to see benchmark results.
"""

import time
import statistics
import random
import sys
import os

# Add the src directory to the path so we can import prefixtrie
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# MUST import pyximport for Cython modules to work properly
import pyximport
pyximport.install()

try:
    from prefixtrie import PrefixTrie
    print("✓ PrefixTrie imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PrefixTrie: {e}")
    sys.exit(1)

try:
    import fuzzysearch
    from fuzzysearch import find_near_matches
    print(f"✓ fuzzysearch imported successfully")
    FUZZYSEARCH_AVAILABLE = True
except ImportError as e:
    print(f"✗ fuzzysearch not available: {e}")
    print("Install with: pip install fuzzysearch")
    FUZZYSEARCH_AVAILABLE = False


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


def generate_target_strings_with_embedded_patterns(patterns: list[str],
                                                  target_count: int = 100,
                                                  target_length: int = 200,
                                                  pattern_ratio: float = 0.7) -> list[str]:
    """Generate target strings with embedded patterns for substring search testing"""
    targets = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for _ in range(target_count):
        target = []
        remaining_length = target_length

        # Decide if this target should contain a pattern
        if random.random() < pattern_ratio and patterns:
            # Choose a random pattern to embed
            pattern = random.choice(patterns)

            # Choose position to embed pattern
            if remaining_length > len(pattern):
                before_length = random.randint(0, remaining_length - len(pattern))
                after_length = remaining_length - before_length - len(pattern)

                # Add random characters before pattern
                before = ''.join(random.choice(alphabet) for _ in range(before_length))
                after = ''.join(random.choice(alphabet) for _ in range(after_length))

                target_str = before + pattern + after
            else:
                # Pattern is too long, just use random string
                target_str = ''.join(random.choice(alphabet) for _ in range(target_length))
        else:
            # Generate completely random string
            target_str = ''.join(random.choice(alphabet) for _ in range(target_length))

        targets.append(target_str)

    return targets


def introduce_errors_to_pattern(pattern: str, error_count: int) -> str:
    """Introduce errors to a pattern for fuzzy matching tests"""
    if error_count == 0:
        return pattern

    pattern_list = list(pattern)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for _ in range(min(error_count, len(pattern))):
        error_type = random.choice(['substitute', 'insert', 'delete'])

        if error_type == 'substitute' and pattern_list:
            pos = random.randint(0, len(pattern_list) - 1)
            pattern_list[pos] = random.choice(alphabet)
        elif error_type == 'insert':
            pos = random.randint(0, len(pattern_list))
            pattern_list.insert(pos, random.choice(alphabet))
        elif error_type == 'delete' and pattern_list:
            pos = random.randint(0, len(pattern_list) - 1)
            pattern_list.pop(pos)

    return ''.join(pattern_list)


def benchmark_prefixtrie_substring(patterns: list[str], targets: list[str], max_corrections: int = 1):
    """Benchmark PrefixTrie substring search performance"""
    print(f"Building PrefixTrie with {len(patterns)} patterns...")
    start_build = time.perf_counter()
    trie = PrefixTrie(patterns, allow_indels=True)
    build_time = time.perf_counter() - start_build

    print(f"Running {len(targets)} substring searches with max {max_corrections} corrections...")
    start_search = time.perf_counter()
    results = []
    for target in targets:
        result = trie.search_substring(target, correction_budget=max_corrections)
        results.append(result)
    search_time = time.perf_counter() - start_search

    return results, build_time, search_time


def benchmark_fuzzysearch(patterns: list[str], targets: list[str], max_dist: int = 1):
    """Benchmark fuzzysearch performance"""
    if not FUZZYSEARCH_AVAILABLE:
        return None, 0, 0

    print(f"Running fuzzysearch on {len(patterns)} patterns with {len(targets)} targets...")

    # fuzzysearch doesn't have a "build" phase like tries, so build_time is 0
    build_time = 0

    start_search = time.perf_counter()
    results = []
    for target in targets:
        best_match = None
        best_dist = max_dist + 1
        best_start = -1
        best_end = -1

        # Search for each pattern in the target
        for pattern in patterns:
            try:
                matches = find_near_matches(pattern, target, max_l_dist=max_dist)
                if matches:
                    # Take the first match (fuzzysearch may return multiple)
                    match = matches[0]
                    if match.dist < best_dist or (match.dist == best_dist and best_match is None):
                        best_match = pattern
                        best_dist = match.dist
                        best_start = match.start
                        best_end = match.end
            except Exception:
                # Handle any fuzzysearch errors
                continue

        if best_match is not None:
            exact = (best_dist == 0)
            results.append((best_match, exact, best_start, best_end))
        else:
            results.append((None, False, -1, -1))

    search_time = time.perf_counter() - start_search

    return results, build_time, search_time


def validate_results_consistency(patterns: list[str], pt_results: list, fs_results: list, test_name: str = ""):
    """Validate that results are consistent between implementations"""
    print(f"  Validating consistency for {test_name}...")

    patterns_set = set(patterns)
    inconsistencies = []

    for i, (pt_result, fs_result) in enumerate(zip(pt_results, fs_results)):
        pt_found, pt_corrections, pt_start, pt_end = pt_result
        fs_found, fs_exact, fs_start, fs_end = fs_result

        # Check if found patterns are valid
        if pt_found is not None and pt_found not in patterns_set:
            inconsistencies.append(f"Index {i}: PrefixTrie found '{pt_found}' not in original patterns")

        if fs_found is not None and fs_found not in patterns_set:
            inconsistencies.append(f"Index {i}: fuzzysearch found '{fs_found}' not in original patterns")

    if inconsistencies:
        print(f"  WARNING: Found {len(inconsistencies)} inconsistencies:")
        for inc in inconsistencies[:3]:  # Show first 3
            print(f"    {inc}")
        if len(inconsistencies) > 3:
            print(f"    ... and {len(inconsistencies) - 3} more")
    else:
        print(f"  ✓ No inconsistencies found")

    return len(inconsistencies) == 0


def run_substring_benchmark(patterns: list[str], targets: list[str], max_corrections: int = 1, num_runs: int = 3, test_name: str = ""):
    """Run a complete substring search benchmark comparison"""
    print(f"\n{'='*60}")
    print(f"SUBSTRING BENCHMARK: {test_name}")
    print(f"Patterns: {len(patterns)}, Targets: {len(targets)}, Max corrections: {max_corrections}")
    print(f"{'='*60}")

    # Benchmark PrefixTrie multiple times
    pt_build_times = []
    pt_search_times = []
    pt_total_times = []
    pt_results = None

    print(f"\nRunning PrefixTrie benchmark ({num_runs} runs)...")
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        results, build_time, search_time = benchmark_prefixtrie_substring(patterns, targets, max_corrections)
        pt_build_times.append(build_time)
        pt_search_times.append(search_time)
        pt_total_times.append(build_time + search_time)
        if pt_results is None:
            pt_results = results

    # Benchmark fuzzysearch multiple times
    fs_build_times = []
    fs_search_times = []
    fs_total_times = []
    fs_results = None

    if FUZZYSEARCH_AVAILABLE:
        print(f"\nRunning fuzzysearch benchmark ({num_runs} runs)...")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            results, build_time, search_time = benchmark_fuzzysearch(patterns, targets, max_corrections)
            fs_build_times.append(build_time)
            fs_search_times.append(search_time)
            fs_total_times.append(build_time + search_time)
            if fs_results is None:
                fs_results = results

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

    if FUZZYSEARCH_AVAILABLE:
        fs_build_avg, fs_build_std = calc_stats(fs_build_times)
        fs_search_avg, fs_search_std = calc_stats(fs_search_times)
        fs_total_avg, fs_total_std = calc_stats(fs_total_times)

        print(f"{'fuzzysearch Build':<20} {fs_build_avg:.4f}s{'':<4} {fs_build_std:.4f}s")
        print(f"{'fuzzysearch Search':<20} {fs_search_avg:.4f}s{'':<4} {fs_search_std:.4f}s")
        print(f"{'fuzzysearch Total':<20} {fs_total_avg:.4f}s{'':<4} {fs_total_std:.4f}s")

        # Calculate speedups
        if pt_search_avg > 0 and fs_search_avg > 0:
            search_speedup = fs_search_avg / pt_search_avg
            total_speedup = fs_total_avg / pt_total_avg

            print(f"\nSpeedup Analysis:")
            print(f"Search speedup: {search_speedup:.2f}x {'(PrefixTrie faster)' if search_speedup > 1 else '(fuzzysearch faster)'}")
            print(f"Total speedup:  {total_speedup:.2f}x {'(PrefixTrie faster)' if total_speedup > 1 else '(fuzzysearch faster)'}")

    # Analyze result quality
    if pt_results and fs_results and FUZZYSEARCH_AVAILABLE:
        pt_found = sum(1 for r in pt_results if r[0] is not None)
        fs_found = sum(1 for r in fs_results if r[0] is not None)

        print(f"\nResult Quality:")
        print(f"PrefixTrie found: {pt_found}/{len(targets)} targets ({pt_found/len(targets)*100:.1f}%)")
        print(f"fuzzysearch found: {fs_found}/{len(targets)} targets ({fs_found/len(targets)*100:.1f}%)")

    # Validate consistency of results
    if pt_results and fs_results and FUZZYSEARCH_AVAILABLE:
        pt_consistent = validate_results_consistency(patterns, pt_results, fs_results, f"{test_name} - Substring Search")

    return {
        'prefixtrie': {
            'build_avg': pt_build_avg,
            'search_avg': pt_search_avg,
            'total_avg': pt_total_avg,
        },
        'fuzzysearch': {
            'build_avg': fs_build_avg if FUZZYSEARCH_AVAILABLE else 0,
            'search_avg': fs_search_avg if FUZZYSEARCH_AVAILABLE else 0,
            'total_avg': fs_total_avg if FUZZYSEARCH_AVAILABLE else 0
        } if FUZZYSEARCH_AVAILABLE else None,
        'patterns_count': len(patterns),
        'targets_count': len(targets)
    }


def main():
    """Run the substring search benchmark suite"""
    print("PrefixTrie vs fuzzysearch Substring Search Benchmark Suite")
    print("=" * 60)

    if not FUZZYSEARCH_AVAILABLE:
        print("Warning: fuzzysearch not available. Only testing PrefixTrie.")
        print("Install fuzzysearch with: pip install fuzzysearch")

    # Set random seed for reproducible results
    random.seed(42)

    # Benchmark scenarios specifically designed for substring search
    scenarios = [
        {
            "name": "Short Patterns - Small Scale",
            "patterns": generate_random_strings(50, 5),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 100, 50, 0.8),
            "max_corrections": 1
        },
        {
            "name": "Short Patterns - Medium Scale",
            "patterns": generate_random_strings(200, 6),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 500, 100, 0.7),
            "max_corrections": 1
        },
        {
            "name": "Medium Patterns - Small Scale",
            "patterns": generate_random_strings(100, 15),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 200, 150, 0.6),
            "max_corrections": 2
        },
        {
            "name": "Medium Patterns - Large Scale",
            "patterns": generate_random_strings(500, 12),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 1000, 200, 0.5),
            "max_corrections": 2
        },
        {
            "name": "Long Patterns",
            "patterns": generate_random_strings(50, 30),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 100, 300, 0.8),
            "max_corrections": 3
        },
        {
            "name": "DNA Sequences - Short",
            "patterns": generate_dna_sequences(100, 10),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 200, 100, 0.7),
            "max_corrections": 1
        },
        {
            "name": "DNA Sequences - Medium",
            "patterns": generate_dna_sequences(200, 20),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 300, 200, 0.6),
            "max_corrections": 2
        },
        {
            "name": "DNA Sequences - Long",
            "patterns": generate_dna_sequences(50, 50),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 100, 500, 0.8),
            "max_corrections": 3
        },
        {
            "name": "Protein Sequences",
            "patterns": generate_protein_sequences(100, 25),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 150, 250, 0.7),
            "max_corrections": 2
        },
        {
            "name": "Realistic Words",
            "patterns": generate_realistic_words(300),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 400, 300, 0.6),
            "max_corrections": 2
        },
        {
            "name": "High Error Rate",
            "patterns": generate_random_strings(100, 10),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 200, 150, 0.7),
            "max_corrections": 4
        },
        {
            "name": "Very Long Targets",
            "patterns": generate_random_strings(50, 8),
            "target_generator": lambda p: generate_target_strings_with_embedded_patterns(p, 50, 1000, 0.9),
            "max_corrections": 2
        }
    ]

    all_results = []

    for scenario in scenarios:
        print(f"\n\nRunning scenario: {scenario['name']}")
        try:
            patterns = scenario['patterns']
            targets = scenario['target_generator'](patterns)

            result = run_substring_benchmark(
                patterns=patterns,
                targets=targets,
                max_corrections=scenario['max_corrections'],
                num_runs=2,  # Reduced runs for faster execution
                test_name=scenario['name']
            )
            result['scenario'] = scenario['name']
            result['max_corrections'] = scenario['max_corrections']
            all_results.append(result)
        except Exception as e:
            print(f"Error in scenario '{scenario['name']}': {e}")
            continue

    # Print comprehensive summary
    print(f"\n\n{'='*100}")
    print("COMPREHENSIVE SUMMARY - SUBSTRING SEARCH BENCHMARKS")
    print("="*100)
    print(f"{'Scenario':<25} {'Patterns':<9} {'Targets':<8} {'Errors':<6} {'PT Search':<10} {'FS Search':<10} {'PT Total':<10} {'FS Total':<10} {'Speedup':<8}")
    print("-" * 100)

    for result in all_results:
        pt_search = result['prefixtrie']['search_avg']
        pt_total = result['prefixtrie']['total_avg']
        patterns_count = result['patterns_count']
        targets_count = result['targets_count']
        max_corrections = result['max_corrections']

        if result['fuzzysearch']:
            fs_search = result['fuzzysearch']['search_avg']
            fs_total = result['fuzzysearch']['total_avg']
            speedup = fs_search / pt_search if pt_search > 0 else 0

            print(f"{result['scenario']:<25} {patterns_count:<9} {targets_count:<8} {max_corrections:<6} "
                  f"{pt_search:.4f}s{'':<2} {fs_search:.4f}s{'':<2} "
                  f"{pt_total:.4f}s{'':<2} {fs_total:.4f}s{'':<2} {speedup:.2f}x")
        else:
            print(f"{result['scenario']:<25} {patterns_count:<9} {targets_count:<8} {max_corrections:<6} "
                  f"{pt_search:.4f}s{'':<2} {'N/A':<10} "
                  f"{pt_total:.4f}s{'':<2} {'N/A':<10} {'N/A':<8}")

    # Calculate overall statistics
    if FUZZYSEARCH_AVAILABLE:
        search_speedups = []
        total_speedups = []

        for result in all_results:
            if result['fuzzysearch'] and result['prefixtrie']['search_avg'] > 0:
                search_speedup = result['fuzzysearch']['search_avg'] / result['prefixtrie']['search_avg']
                total_speedup = result['fuzzysearch']['total_avg'] / result['prefixtrie']['total_avg']

                search_speedups.append(search_speedup)
                total_speedups.append(total_speedup)

        if search_speedups:
            avg_search_speedup = statistics.mean(search_speedups)
            avg_total_speedup = statistics.mean(total_speedups)
            median_search_speedup = statistics.median(search_speedups)
            median_total_speedup = statistics.median(total_speedups)

            print(f"\nOVERALL PERFORMANCE ANALYSIS:")
            print(f"Average search speedup: {avg_search_speedup:.2f}x {'(PrefixTrie faster)' if avg_search_speedup > 1 else '(fuzzysearch faster)'}")
            print(f"Median search speedup:  {median_search_speedup:.2f}x {'(PrefixTrie faster)' if median_search_speedup > 1 else '(fuzzysearch faster)'}")
            print(f"Average total speedup:  {avg_total_speedup:.2f}x {'(PrefixTrie faster)' if avg_total_speedup > 1 else '(fuzzysearch faster)'}")
            print(f"Median total speedup:   {median_total_speedup:.2f}x {'(PrefixTrie faster)' if median_total_speedup > 1 else '(fuzzysearch faster)'}")

    print("\nSubstring search benchmark complete!")


if __name__ == "__main__":
    main()
