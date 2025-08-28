import pickle
import multiprocessing as mp
import tempfile
import os
import pytest
import platform
import io
import copy
import gc
import weakref
import pyximport
pyximport.install()
from prefixtrie import PrefixTrie, create_shared_trie, load_shared_trie


# Module-level worker functions for multiprocessing tests
# These need to be at module level to be picklable

def worker_search(trie, query, correction_budget=0):
    """Worker function that uses a PrefixTrie"""
    return trie.search(query, correction_budget)


def fuzzy_worker(trie, query, budget):
    """Worker function for fuzzy search"""
    return trie.search(query, correction_budget=budget)


def search_worker(args):
    """Worker function for map operation"""
    trie, query = args
    return trie.search(query)


def starmap_worker(trie, query, budget):
    """Worker function for starmap operation"""
    return trie.search(query, correction_budget=budget)


def batch_search_worker(args):
    """Worker that performs multiple searches"""
    trie, queries = args
    results = []
    for query in queries:
        result = trie.search(query)
        results.append(result)
    return results


def config_worker(args):
    """Worker that creates and uses a trie with specific config"""
    entries, allow_indels, query, budget = args
    trie = PrefixTrie(entries, allow_indels=allow_indels)
    return trie.search(query, correction_budget=budget)


def large_trie_worker(args):
    """Worker for large trie operations"""
    trie, start_idx, end_idx = args
    count = 0
    for i in range(start_idx, end_idx):
        query = f"entry_{i:04d}"
        result, corrections = trie.search(query)
        if result == query and corrections == 0:
            count += 1
    return count


def simple_worker(trie):
    """Simple worker for basic multiprocessing test"""
    return trie.search("world")


def shared_memory_worker(shared_memory_name):
    """Worker that loads from shared memory and performs searches"""
    trie = load_shared_trie(shared_memory_name)
    results = []
    # Search for entries that actually exist in the test trie
    for query in ["shared", "memory", "multiprocessing"]:
        result = trie.search(query)
        results.append(result)
    return results


def shared_memory_fuzzy_worker(args):
    """Worker for fuzzy search using shared memory"""
    shared_memory_name, query, budget = args
    trie = load_shared_trie(shared_memory_name)
    return trie.search(query, correction_budget=budget)


def shared_memory_batch_worker(args):
    """Worker that performs batch operations on shared memory trie"""
    shared_memory_name, queries = args
    trie = load_shared_trie(shared_memory_name)
    results = []
    for query in queries:
        result = trie.search(query)
        results.append(result)
    return results


# Module-level class for pickle testing (needs to be at module level to be picklable)
class TrieWrapper:
    """Wrapper class for testing pickle interaction with custom objects containing tries"""
    def __init__(self, trie):
        self.trie = trie
        self.metadata = {"created_by": "test", "version": 1.0}
        self.search_count = 0

    def search(self, query, **kwargs):
        self.search_count += 1
        return self.trie.search(query, **kwargs)

    def __getstate__(self):
        # Custom pickle state
        return {
            'trie': self.trie,
            'metadata': self.metadata,
            'search_count': self.search_count
        }

    def __setstate__(self, state):
        # Custom unpickle state
        self.trie = state['trie']
        self.metadata = state['metadata']
        self.search_count = state['search_count']


class TestPrefixTriePickle:
    """Test pickle compatibility of PrefixTrie"""

    def test_basic_pickle_roundtrip(self):
        """Test basic pickle serialization and deserialization"""
        entries = ["hello", "world", "test", "python"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Serialize
        pickled_data = pickle.dumps(trie)

        # Deserialize
        restored_trie = pickle.loads(pickled_data)

        # Verify the restored trie works correctly
        assert len(restored_trie) == len(trie)
        assert restored_trie.allow_indels == trie.allow_indels

        # Test functionality
        for entry in entries:
            result, corrections = restored_trie.search(entry)
            assert result == entry
            assert corrections == 0
            assert entry in restored_trie
            assert restored_trie[entry] == entry

    def test_pickle_with_fuzzy_search(self):
        """Test that fuzzy search works after pickle roundtrip"""
        entries = ["algorithm", "logarithm", "rhythm"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Pickle and unpickle
        pickled_data = pickle.dumps(trie)
        restored_trie = pickle.loads(pickled_data)

        # Test fuzzy search
        result, corrections = restored_trie.search("algrothm", correction_budget=2)
        assert result == "algorithm"
        assert corrections == 2

        result, corrections = restored_trie.search("rythem", correction_budget=2)
        assert result == "rhythm"
        assert corrections > 0

    def test_pickle_different_protocols(self):
        """Test pickle with different protocol versions"""
        entries = ["cat", "car", "card", "care"]
        trie = PrefixTrie(entries, allow_indels=False)

        # Test with different pickle protocols
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            pickled_data = pickle.dumps(trie, protocol=protocol)
            restored_trie = pickle.loads(pickled_data)

            # Verify functionality
            assert len(restored_trie) == len(trie)
            for entry in entries:
                result, corrections = restored_trie.search(entry)
                assert result == entry
                assert corrections == 0

    def test_pickle_empty_trie(self):
        """Test pickle with empty trie"""
        trie = PrefixTrie([])

        pickled_data = pickle.dumps(trie)
        restored_trie = pickle.loads(pickled_data)

        assert len(restored_trie) == 0
        result, corrections = restored_trie.search("anything")
        assert result is None
        assert corrections == -1

    def test_pickle_large_trie(self):
        """Test pickle with large trie"""
        entries = [f"entry_{i:04d}" for i in range(1000)]
        trie = PrefixTrie(entries, allow_indels=True)

        pickled_data = pickle.dumps(trie)
        restored_trie = pickle.loads(pickled_data)

        assert len(restored_trie) == len(trie)

        # Test a few entries
        test_entries = [entries[0], entries[500], entries[999]]
        for entry in test_entries:
            result, corrections = restored_trie.search(entry)
            assert result == entry
            assert corrections == 0

    def test_pickle_to_file(self):
        """Test pickle serialization to file"""
        entries = ["file", "test", "pickle", "serialization"]
        trie = PrefixTrie(entries, allow_indels=True)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(trie, f)
            filename = f.name

        try:
            with open(filename, 'rb') as f:
                restored_trie = pickle.load(f)

            # Verify functionality
            assert len(restored_trie) == len(trie)
            for entry in entries:
                result, corrections = restored_trie.search(entry)
                assert result == entry
                assert corrections == 0
        finally:
            os.unlink(filename)

    def test_pickle_state_consistency(self):
        """Test that all internal state is properly preserved during pickle"""
        entries = ["test", "testing", "tester", "tea", "team"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Get initial state
        initial_len = len(trie)
        initial_allow_indels = trie.allow_indels
        initial_entries = list(trie)

        # Test various search operations before pickling
        search_results_before = []
        for entry in entries:
            result = trie.search(entry)
            search_results_before.append(result)

        # Pickle and unpickle
        pickled_data = pickle.dumps(trie)
        restored_trie = pickle.loads(pickled_data)

        # Verify state consistency
        assert len(restored_trie) == initial_len
        assert restored_trie.allow_indels == initial_allow_indels
        assert set(restored_trie) == set(initial_entries)

        # Test that search results are identical
        search_results_after = []
        for entry in entries:
            result = restored_trie.search(entry)
            search_results_after.append(result)

        assert search_results_before == search_results_after

    def test_pickle_memory_efficiency(self):
        """Test that pickle doesn't create memory leaks or excessive objects"""
        entries = [f"mem_test_{i}" for i in range(100)]
        trie = PrefixTrie(entries, allow_indels=False)

        # Track object count before pickle operations
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform multiple pickle/unpickle cycles
        for _ in range(10):
            pickled_data = pickle.dumps(trie)
            restored_trie = pickle.loads(pickled_data)
            # Verify functionality
            assert len(restored_trie) == len(trie)
            del restored_trie

        # Check for memory leaks
        gc.collect()
        final_objects = len(gc.get_objects())

        # Allow some tolerance for Python's object management
        assert final_objects - initial_objects < 50

    def test_pickle_serialization_formats(self):
        """Test pickle with different serialization formats and compression"""
        entries = ["format", "test", "compression", "serialization"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test different pickle protocols
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            # Test binary format
            binary_data = pickle.dumps(trie, protocol=protocol)
            restored_binary = pickle.loads(binary_data)

            assert len(restored_binary) == len(trie)
            assert restored_binary.allow_indels == trie.allow_indels

            # Test with BytesIO
            buffer = io.BytesIO()
            pickle.dump(trie, buffer, protocol=protocol)
            buffer.seek(0)
            restored_buffer = pickle.load(buffer)

            assert len(restored_buffer) == len(trie)
            assert restored_buffer.allow_indels == trie.allow_indels

    def test_pickle_circular_references(self):
        """Test pickle behavior with circular references (if any)"""
        entries = ["circular", "reference", "test"]
        trie = PrefixTrie(entries, allow_indels=False)

        # Create a container with circular reference
        container = {"trie": trie, "self": None}
        container["self"] = container
        container["trie_copy"] = trie  # Same trie referenced twice

        # Should be able to pickle container with trie
        pickled_data = pickle.dumps(container)
        restored_container = pickle.loads(pickled_data)

        # Verify the trie works in the restored container
        assert restored_container["self"] is restored_container
        assert len(restored_container["trie"]) == len(trie)
        assert len(restored_container["trie_copy"]) == len(trie)

        # Both trie references should work
        result, corrections = restored_container["trie"].search("circular")
        assert result == "circular" and corrections == 0

        result, corrections = restored_container["trie_copy"].search("reference")
        assert result == "reference" and corrections == 0

    def test_pickle_with_custom_objects(self):
        """Test pickle interaction with custom objects containing tries"""
        entries = ["custom", "object", "wrapper"]

        # Use the module-level TrieWrapper class (defined at the top of the file)
        trie = PrefixTrie(entries, allow_indels=True)
        wrapper = TrieWrapper(trie)

        # Use the wrapper
        wrapper.search("custom")
        wrapper.search("object")
        initial_count = wrapper.search_count

        # Pickle and unpickle
        pickled_data = pickle.dumps(wrapper)
        restored_wrapper = pickle.loads(pickled_data)

        # Verify wrapper state
        assert restored_wrapper.search_count == initial_count
        assert restored_wrapper.metadata == wrapper.metadata

        # Verify trie functionality
        result, corrections = restored_wrapper.search("wrapper")
        assert result == "wrapper" and corrections == 0
        assert restored_wrapper.search_count == initial_count + 1

    def test_pickle_error_handling(self):
        """Test pickle error handling and recovery"""
        entries = ["error", "handling", "test"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test with corrupted pickle data
        valid_pickled_data = pickle.dumps(trie)

        # Corrupt the data
        corrupted_data = valid_pickled_data[:-10] + b"corrupted"

        with pytest.raises((pickle.PickleError, EOFError, ValueError)):
            pickle.loads(corrupted_data)

        # Test with truncated data
        truncated_data = valid_pickled_data[:len(valid_pickled_data)//2]

        with pytest.raises((pickle.PickleError, EOFError)):
            pickle.loads(truncated_data)

        # Test that original trie still works after failed unpickling attempts
        result, corrections = trie.search("error")
        assert result == "error" and corrections == 0

    def test_pickle_version_compatibility(self):
        """Test pickle data compatibility across versions"""
        entries = ["version", "compatibility", "test"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Create pickle data with different protocols
        pickle_data_by_protocol = {}
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            pickle_data_by_protocol[protocol] = pickle.dumps(trie, protocol=protocol)

        # Verify all protocol versions can be loaded
        for protocol, data in pickle_data_by_protocol.items():
            restored_trie = pickle.loads(data)
            assert len(restored_trie) == len(trie)
            assert restored_trie.allow_indels == trie.allow_indels

            # Test functionality
            for entry in entries:
                result, corrections = restored_trie.search(entry)
                assert result == entry and corrections == 0

    def test_pickle_thread_safety_simulation(self):
        """Test pickle in scenarios that simulate thread safety concerns"""
        entries = ["thread", "safety", "simulation"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Simulate concurrent pickle operations
        pickled_copies = []
        for i in range(10):
            # Create multiple pickled copies
            pickled_data = pickle.dumps(trie)
            pickled_copies.append(pickled_data)

        # Restore all copies and verify they work independently
        restored_tries = []
        for pickled_data in pickled_copies:
            restored_trie = pickle.loads(pickled_data)
            restored_tries.append(restored_trie)

        # Verify all restored tries work correctly
        for restored_trie in restored_tries:
            assert len(restored_trie) == len(trie)
            for entry in entries:
                result, corrections = restored_trie.search(entry)
                assert result == entry and corrections == 0

    def test_pickle_with_deepcopy(self):
        """Test interaction between pickle and deepcopy"""
        entries = ["deepcopy", "interaction", "test"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test deepcopy
        copied_trie = copy.deepcopy(trie)
        assert len(copied_trie) == len(trie)

        # Test pickle of deepcopied trie
        pickled_copy = pickle.dumps(copied_trie)
        restored_copy = pickle.loads(pickled_copy)

        # Verify all versions work
        for test_trie in [trie, copied_trie, restored_copy]:
            for entry in entries:
                result, corrections = test_trie.search(entry)
                assert result == entry and corrections == 0

    def test_pickle_performance_large_data(self):
        """Test pickle performance with large data sets"""
        # Create a large trie
        entries = [f"performance_test_entry_{i:06d}" for i in range(1000)]
        trie = PrefixTrie(entries, allow_indels=False)

        # Measure pickle performance
        import time
        start_time = time.time()
        pickled_data = pickle.dumps(trie)
        pickle_time = time.time() - start_time

        # Measure unpickle performance
        start_time = time.time()
        restored_trie = pickle.loads(pickled_data)
        unpickle_time = time.time() - start_time

        # Verify correctness
        assert len(restored_trie) == len(trie)

        # Test a few random entries
        test_indices = [0, 100, 500, 999]
        for i in test_indices:
            entry = entries[i]
            result, corrections = restored_trie.search(entry)
            assert result == entry and corrections == 0

        # Basic performance assertions (these are quite lenient)
        assert pickle_time < 10.0  # Should pickle in under 10 seconds
        assert unpickle_time < 10.0  # Should unpickle in under 10 seconds
        assert len(pickled_data) > 1000  # Should have substantial size

class TestPrefixTrieMultiprocessing:
    """Test multiprocessing compatibility of PrefixTrie"""

    def test_multiprocessing_worker_function(self):
        """Test that PrefixTrie can be used in multiprocessing worker functions"""

        entries = ["multiprocessing", "test", "worker", "function"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Test with multiprocessing
        with mp.Pool(processes=2) as pool:
            # Test exact searches
            results = []
            for entry in entries:
                result = pool.apply_async(worker_search, (trie, entry))
                results.append(result)

            for i, result in enumerate(results):
                found, corrections = result.get(timeout=10)
                assert found == entries[i]
                assert corrections == 0

    def test_multiprocessing_fuzzy_search(self):
        """Test fuzzy search in multiprocessing context"""

        entries = ["parallel", "processing", "fuzzy", "search"]
        trie = PrefixTrie(entries, allow_indels=True)

        with mp.Pool(processes=2) as pool:
            # Test fuzzy searches
            test_cases = [
                ("paralel", 1, "parallel", 1),
                ("procesing", 1, "processing", 1),
                ("fuzy", 1, "fuzzy", 1),
                ("serch", 1, "search", 1),
            ]

            results = []
            for query, budget, expected, expected_corrections in test_cases:
                result = pool.apply_async(fuzzy_worker, (trie, query, budget))
                results.append((result, expected, expected_corrections))

            for result, expected, expected_corrections in results:
                found, corrections = result.get(timeout=10)
                assert found == expected
                assert corrections == expected_corrections

    def test_multiprocessing_map(self):
        """Test using PrefixTrie with multiprocessing.Pool.map"""

        entries = ["map", "function", "test", "multiprocessing"]
        trie = PrefixTrie(entries, allow_indels=False)

        # Prepare arguments for map
        queries = ["map", "function", "test", "nonexistent"]
        args_list = [(trie, query) for query in queries]

        with mp.Pool(processes=2) as pool:
            results = pool.map(search_worker, args_list)

        # Verify results
        expected = [
            ("map", 0),
            ("function", 0),
            ("test", 0),
            (None, -1)
        ]

        assert results == expected

    def test_multiprocessing_starmap(self):
        """Test using PrefixTrie with multiprocessing.Pool.starmap"""

        entries = ["starmap", "operation", "testing"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Prepare arguments for starmap
        args_list = [
            (trie, "starmap", 0),
            (trie, "operation", 0),
            (trie, "testin", 1),  # missing 'g'
            (trie, "starmep", 1),  # 'a'->'e' substitution
        ]

        with mp.Pool(processes=2) as pool:
            results = pool.starmap(starmap_worker, args_list)

        # Verify results
        expected_found = ["starmap", "operation", "testing", "starmap"]
        expected_corrections = [0, 0, 1, 1]

        for i, (found, corrections) in enumerate(results):
            assert found == expected_found[i]
            assert corrections == expected_corrections[i]

    def test_shared_trie_multiple_processes(self):
        """Test sharing the same trie across multiple processes"""

        entries = ["shared", "trie", "multiple", "processes", "testing"]
        trie = PrefixTrie(entries, allow_indels=False)

        # Split queries among processes
        all_queries = ["shared", "trie", "multiple", "processes", "testing"]
        query_batches = [
            ["shared", "trie"],
            ["multiple", "processes"],
            ["testing"]
        ]

        args_list = [(trie, batch) for batch in query_batches]

        with mp.Pool(processes=len(query_batches)) as pool:
            batch_results = pool.map(batch_search_worker, args_list)

        # Flatten results and verify
        all_results = []
        for batch in batch_results:
            all_results.extend(batch)

        expected = [(query, 0) for query in all_queries]
        assert all_results == expected

    def test_process_with_different_trie_configs(self):
        """Test processes with different trie configurations"""

        configs = [
            (["exact", "match"], False, "exact", 0),
            (["fuzzy", "match"], True, "fuzy", 1),  # missing 'z'
            (["another", "test"], True, "anther", 1),  # missing 'o'
        ]

        with mp.Pool(processes=len(configs)) as pool:
            results = pool.map(config_worker, configs)

        expected = [
            ("exact", 0),
            ("fuzzy", 1),
            ("another", 1)
        ]

        assert results == expected

    @pytest.mark.skipif(mp.get_start_method() == 'spawn',
                       reason="This test requires 'fork' start method for shared memory")
    def test_large_trie_multiprocessing(self):
        """Test multiprocessing with large trie (if fork method available)"""

        # Create large trie
        entries = [f"entry_{i:04d}" for i in range(1000)]
        trie = PrefixTrie(entries, allow_indels=False)

        # Split work among processes
        chunk_size = 250
        args_list = [
            (trie, i, min(i + chunk_size, 1000))
            for i in range(0, 1000, chunk_size)
        ]

        with mp.Pool(processes=2) as pool:
            counts = pool.map(large_trie_worker, args_list)

        # Verify all entries were found
        total_found = sum(counts)
        assert total_found == 1000


class TestPrefixTrieSharedMemory:
    """Test shared memory functionality of PrefixTrie"""

    def test_basic_shared_memory_creation(self):
        """Test basic shared memory creation and loading"""
        entries = ["hello", "world", "test", "python"]
        trie = PrefixTrie(entries, allow_indels=True)

        # Create shared memory
        shm_name = trie.create_shared_memory()

        try:
            # Load from shared memory
            loaded_trie = load_shared_trie(shm_name)

            # Verify functionality
            assert len(loaded_trie) == len(trie)
            assert loaded_trie.allow_indels == trie.allow_indels

            for entry in entries:
                result, corrections = loaded_trie.search(entry)
                assert result == entry
                assert corrections == 0
        finally:
            trie.cleanup_shared_memory()

    def test_create_shared_trie_convenience_function(self):
        """Test the convenience function for creating shared tries"""
        entries = ["shared", "memory", "test"]

        trie, shm_name = create_shared_trie(entries, allow_indels=True)

        try:
            # Load from shared memory using the name
            loaded_trie = load_shared_trie(shm_name)

            # Verify functionality
            assert len(loaded_trie) == len(trie)
            assert loaded_trie.allow_indels == trie.allow_indels

            for entry in entries:
                result, corrections = loaded_trie.search(entry)
                assert result == entry
                assert corrections == 0
        finally:
            trie.cleanup_shared_memory()

    def test_shared_memory_with_fuzzy_search(self):
        """Test fuzzy search works with shared memory"""
        entries = ["algorithm", "logarithm", "rhythm"]
        trie, shm_name = create_shared_trie(entries, allow_indels=True)

        try:
            loaded_trie = load_shared_trie(shm_name)

            # Test fuzzy search
            result, corrections = loaded_trie.search("algrothm", correction_budget=2)
            assert result == "algorithm"
            assert corrections == 2

            result, corrections = loaded_trie.search("rythem", correction_budget=2)
            assert result == "rhythm"
            assert corrections > 0
        finally:
            trie.cleanup_shared_memory()

    def test_shared_memory_multiprocessing_basic(self):
        """Test basic multiprocessing with shared memory"""
        entries = ["shared", "memory", "multiprocessing"]
        trie, shm_name = create_shared_trie(entries, allow_indels=False)

        try:
            with mp.Pool(processes=2) as pool:
                # Test multiple workers accessing shared memory
                results = []
                for _ in range(3):
                    result = pool.apply_async(shared_memory_worker, (shm_name,))
                    results.append(result)

                # Verify all workers got correct results
                expected = [("shared", 0), ("memory", 0), ("multiprocessing", 0)]
                for result in results:
                    worker_results = result.get(timeout=10)
                    assert worker_results == expected
        finally:
            trie.cleanup_shared_memory()

    def test_shared_memory_fuzzy_multiprocessing(self):
        """Test fuzzy search with shared memory across processes"""
        entries = ["parallel", "processing", "fuzzy", "search"]
        trie, shm_name = create_shared_trie(entries, allow_indels=True)

        try:
            with mp.Pool(processes=2) as pool:
                test_cases = [
                    (shm_name, "paralel", 1, "parallel", 1),
                    (shm_name, "procesing", 1, "processing", 1),
                    (shm_name, "fuzy", 1, "fuzzy", 1),
                    (shm_name, "serch", 1, "search", 1),
                ]

                results = []
                for args in test_cases:
                    result = pool.apply_async(shared_memory_fuzzy_worker, (args[:3],))
                    results.append((result, args[3], args[4]))

                for result, expected, expected_corrections in results:
                    found, corrections = result.get(timeout=10)
                    assert found == expected
                    assert corrections == expected_corrections
        finally:
            trie.cleanup_shared_memory()

    def test_shared_memory_large_trie_performance(self):
        """Test shared memory with large trie for performance comparison"""
        # Create a large trie
        entries = [f"entry_{i:04d}" for i in range(1000)]
        trie, shm_name = create_shared_trie(entries, allow_indels=False)

        try:
            with mp.Pool(processes=2) as pool:
                # Split queries among workers
                query_batches = [
                    entries[:250],
                    entries[250:500],
                    entries[500:750],
                    entries[750:1000]
                ]

                args_list = [(shm_name, batch) for batch in query_batches]

                results = pool.map(shared_memory_batch_worker, args_list)

                # Flatten and verify results
                all_results = []
                for batch_result in results:
                    all_results.extend(batch_result)

                expected_results = [(entry, 0) for entry in entries]
                assert all_results == expected_results
        finally:
            trie.cleanup_shared_memory()

    def test_shared_memory_concurrent_access(self):
        """Test concurrent access to shared memory from multiple processes"""
        entries = ["concurrent", "access", "test", "shared", "memory"]
        trie, shm_name = create_shared_trie(entries, allow_indels=True)

        try:
            with mp.Pool(processes=4) as pool:
                # Have multiple processes access the same shared memory simultaneously
                tasks = []
                for _ in range(10):  # 10 concurrent tasks
                    task = pool.apply_async(shared_memory_worker, (shm_name,))
                    tasks.append(task)

                # Verify all tasks complete successfully
                # shared_memory_worker searches for ["shared", "memory", "multiprocessing"]
                # but our trie has ["concurrent", "access", "test", "shared", "memory"]
                expected = [("shared", 0), ("memory", 0), (None, -1)]  # "multiprocessing" not in trie

                results = [task.get(timeout=10) for task in tasks]

                for result in results:
                    assert result == expected
        finally:
            trie.cleanup_shared_memory()

    def test_shared_memory_with_custom_name(self):
        """Test shared memory with custom name"""
        entries = ["custom", "name", "test"]
        trie = PrefixTrie(entries, allow_indels=False)

        custom_name = "test_custom_shared_memory"
        shm_name = trie.create_shared_memory(name=custom_name)

        try:
            assert shm_name == custom_name

            loaded_trie = load_shared_trie(custom_name)

            for entry in entries:
                result, corrections = loaded_trie.search(entry)
                assert result == entry
                assert corrections == 0
        finally:
            trie.cleanup_shared_memory()

    def test_shared_memory_cleanup(self):
        """Test that shared memory is properly cleaned up"""
        entries = ["cleanup", "test"]
        trie, shm_name = create_shared_trie(entries)

        # Verify shared memory exists
        loaded_trie = load_shared_trie(shm_name)
        result, corrections = loaded_trie.search("cleanup")
        assert result == "cleanup"

        # Clean up
        trie.cleanup_shared_memory()

        # Verify shared memory is no longer accessible
        # On Windows, shared memory might still be accessible after unlink()
        # due to platform differences in shared memory behavior
        if platform.system() == "Windows":
            # On Windows, we can't guarantee the shared memory is immediately inaccessible
            # Just verify that cleanup was called without error
            assert trie._shared_memory is None
            assert trie._is_shared_owner is False
        else:
            # On Unix/Linux, shared memory should be inaccessible after cleanup
            with pytest.raises(RuntimeError):
                load_shared_trie(shm_name)

    def test_shared_memory_error_handling(self):
        """Test error handling for shared memory operations"""
        # Test loading from non-existent shared memory
        with pytest.raises(RuntimeError):
            load_shared_trie("non_existent_shared_memory")

        # Test creating shared memory with same name twice
        entries = ["error", "handling"]
        trie1 = PrefixTrie(entries)

        try:
            name1 = trie1.create_shared_memory(name="duplicate_name")

            trie2 = PrefixTrie(entries)
            # This should fail because the name already exists
            with pytest.raises(RuntimeError):
                trie2.create_shared_memory(name="duplicate_name")
        finally:
            trie1.cleanup_shared_memory()

    def test_shared_memory_backwards_compatibility(self):
        """Test that shared memory doesn't break regular pickle compatibility"""
        entries = ["backwards", "compatibility", "test"]

        # Create regular trie
        trie = PrefixTrie(entries, allow_indels=True)

        # Should still work with regular pickle
        pickled_data = pickle.dumps(trie)
        restored_trie = pickle.loads(pickled_data)

        # Verify functionality
        assert len(restored_trie) == len(trie)
        assert restored_trie.allow_indels == trie.allow_indels

        for entry in entries:
            result, corrections = restored_trie.search(entry)
            assert result == entry
            assert corrections == 0


# Update the shared_memory_worker to use the correct entries
def shared_memory_concurrent_worker(shared_memory_name):
    """Worker for concurrent access test"""
    trie = load_shared_trie(shared_memory_name)
    results = []
    for query in ["concurrent", "access", "test"]:
        result = trie.search(query)
        results.append(result)
    return results


if __name__ == "__main__":
    # Run a quick smoke test for pickle functionality
    print("Running pickle smoke test...")

    # Basic pickle test
    trie = PrefixTrie(["hello", "world"], allow_indels=True)
    pickled = pickle.dumps(trie)
    restored = pickle.loads(pickled)

    result, corrections = restored.search("hello")
    assert result == "hello" and corrections == 0

    result, corrections = restored.search("helo", correction_budget=1)
    assert result == "hello" and corrections == 1

    print("Pickle smoke test passed!")

    # Basic multiprocessing test
    print("Running multiprocessing smoke test...")

    try:
        with mp.Pool(processes=1) as pool:
            result = pool.apply(simple_worker, (trie,))
            assert result == ("world", 0)
        print("Multiprocessing smoke test passed!")
    except Exception as e:
        print(f"Multiprocessing test failed: {e}")
        print("This may be due to the multiprocessing start method. Run full tests with pytest.")

    print("Run 'pytest test/test_pickle.py' for full test suite.")
