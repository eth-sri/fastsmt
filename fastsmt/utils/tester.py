"""
Copyright 2018 Software Reliability Lab, ETH Zurich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import os
import traceback
import json
import pickle
import logging
import tempfile
import time

from fastsmt.utils.parallel_run import run_ordered_tasks_and_merge_outputs, get_merge_files_callback
from fastsmt.language.objects import *
import fastsmt.utils.test as utils_test

def exec_test(test, out_file):
    try:
        test.run()
    except:
        print("Unexpected error:", sys.exc_info())
        traceback.print_exc(file=sys.stdout)
    finally:
        with open(out_file, "a") as f:
            f.write("%s\n" % (json.dumps(test.to_json())))


class BenchmarkGoalTester(object):

    def __init__(self, only_cached=False, max_cache_size=10**10, out_file = None, num_threads=1, tmp_dir='tmp/'):
        """ Initializes object of type BenchmarkGoalTester.

        Arguments:
            only_cached - if True then only cached tests are run
            max_cache_size - maximum size of the cache
            out_file - file where results should be stored
            num_threads - number of threads to use during parallel evaluation
            tmp_dir - temporary directory where SMT formulas should be saved
        """
        self.cache = {}
        self.only_cached = only_cached
        self.max_cache_size = max_cache_size
        self.num_threads = num_threads
        self.tmp_dir = tmp_dir
        self.tot_cached = 0

        self.log = logging.getLogger('BenchmarkGoalTester')

        self.cache_path = os.path.join(self.tmp_dir, 'cache')
        if not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path)

        self.cache_dir = tempfile.TemporaryDirectory(dir=self.cache_path, suffix='.d')
        self.out_file = out_file
        self.data = []

    def cleanup(self):
        for smt_file in self.cache:
            for task in self.cache[smt_file].values():
                task.out_file = None
        self.cache_dir.cleanup()
        self.cache_dir = tempfile.TemporaryDirectory(dir=self.cache_path, suffix='.d')

    def close(self):
        assert self.cache_dir is not None, 'Tester already closed!'
        self.log.debug('Closing tester!')
        self.save()
        self.cache_dir.cleanup()
        self.cache_dir = None

    def get_best(self, file):
        file_cache = self.cache[file]

        best_task = None
        for strategy, task in file_cache.items():
            if (task.res == 'sat' or task.res == 'unsat') and (best_task is None or best_task.res > task.res):
                best_task = task
        return best_task

    def find_cached_prefix(self, file, original_strategy):
        """
        Given file and strategy, it finds longest prefix of the strategy such that
        application of that prefix to the file is cached.
        """
        if file not in self.cache:
            return None

        file_cache = self.cache[file]
        parent_strategy = original_strategy

        while parent_strategy:
            if str(parent_strategy) in file_cache:
                return file_cache[str(parent_strategy)]

            tactics = get_tactics(parent_strategy)
            if len(tactics) > 1 and str(tactics[0]) == 'Tactic(skip)':
                new_strat = make_strategy(tactics[1:])
                if str(new_strat) in file_cache:
                    cached_test = file_cache[str(new_strat)]
                    cached_test.strat = parent_strategy
                    return cached_test

            parent_strategy, removed_tactic = shorten_strategy(parent_strategy)
        return None

    def get_cached_test(self, test):
        """ Returns cache entry which corresponds to test. """
        if test.file not in self.cache:
            return None

        tactics = get_tactics(test.strat)
        cached_test = self.cache[test.file].get(str(test.strat))

        # In this case first skip is irrelevant so we can search cache without it
        if len(tactics) > 1 and str(tactics[0]) == 'Tactic(skip)':
            new_strat = make_strategy(tactics[1:])
            # print('Looking for cache')
            # print(test.file)
            # print(new_strat)
            cached_test = self.cache[test.file].get(str(new_strat))
            if cached_test is not None:
                cached_test.strat = test.strat
            # print('---> ',str(test))
            # print('-> ',str(cached_test))
            # assert cached_test is not None

        if cached_test is not None:
            self.log.debug('Cached Test:' + str(cached_test))
            self.log.debug('Test:' + str(test))
        return cached_test

    def add_test(self, test):
        """ Adds test to cache. """
        if isinstance(test, utils_test.IncrementalBenchmarkTest):
            assert test.partial_tasks is not None
            assert len(test.partial_tasks) > 0
            for partial_task in test.partial_tasks:
                self.add_test(utils_test.BenchmarkGoalTest.from_json(partial_task))
            test.partial_tasks = []
        self.tot_cached += 1
        if test.file not in self.cache:
            self.cache[test.file] = {}

        cached_test = self.cache[test.file].get(str(test.strat))
        if cached_test is None or cached_test.timeout < test.timeout:
            self.cache[test.file][str(test.strat)] = test

    def load_results(self, file):
        """ Given file such that each line corresponds to JSON of test, returns list of tests. """
        results = []
        with open(file, "r") as f:
            for line in f:
                line_json = json.loads(line)
                if line_json['kind'] == 'BenchmarkGoalTest':
                    results.append(utils_test.BenchmarkGoalTest.from_json(line_json))
                elif line_json['kind'] == 'BenchmarkTest':
                    results.append(utils_test.BenchmarkTest.from_json(line_json))
                else:
                    results.append(utils_test.IncrementalBenchmarkTest.from_json(line_json))
        return results

    def save_cache(self):
        self.log.info('Saving cache to: ' + str(self.out_file))

        if self.out_file is None:
            return
        
        with open(self.out_file, 'w') as f:
            for smt_file in self.cache:
                for task in self.cache[smt_file].values():
                    # if isinstance(task, IncrementalBenchmarkTest):
                    #     continue
                    value = {
                        'task': task.to_json(),
                        'pruned': None,
                        'solved': None,
                        'status': None
                    }
                    f.write(json.dumps(value) + '\n')


    def load_cache(self, file):
        """ Loads cache from a file. """
        self.log.info('Loading cache...')

        if not os.path.isfile(file):
            self.log.info('Cache file does not exist, not loading!')
            return

        num_tests = 0
        with open(file, "r") as f:
            for line in f:
                try:
                    values = json.loads(line)
                    test = utils_test.BenchmarkGoalTest.from_json(values['task'])
                    test.out_file = None

                    num_tests += 1
                    self.add_test(test)
                except json.decoder.JSONDecodeError:
                    self.log.warning('Line can not be read by JSON decoder!')


        self.log.info('Done. Found %d Tests' % (num_tests))

    def assign_cached_result(self, test):
        if test.res == 'sat' or test.res == 'unsat' or test.res == 'fail':
            return test
        """ Given test, corresponding entry from cache is found and test is updated with cached result of evaluation. """
        cached_test = self.get_cached_test(test)
        if cached_test is None:
            if self.only_cached:
                test.res = 'unknown'
                test.rlimit = 0
                test.runtime = 0
                test.probes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            return

        # if the result is sat/unsat then we return the result directly
        # if the cached timeout is higher and the result is unknown then lower timeout does not change anything
        if cached_test.is_solved() or cached_test.timeout >= test.timeout:
            test.res = cached_test.res
            test.rlimit = cached_test.rlimit
            test.runtime = cached_test.runtime
            test.in_hash = cached_test.in_hash
            test.out_hash = cached_test.out_hash
            test.out_file = cached_test.out_file
            test.probes = cached_test.probes
            test.kind = cached_test.kind
            return

        return

    def compute_cached_outputs(self, test):
        """ Recomputes output file of cached prefix of test. """
        cached_goal = self.find_cached_prefix(test.file, str(test.strat))
        assert not (cached_goal is not None and str(cached_goal.strat) == str(test.strat) and cached_goal.timeout >= test.timeout)

        if test.out_file is None:
            assert self.cache_dir
            tmp_file = tempfile.NamedTemporaryFile(
                suffix='.qwe', dir=self.cache_dir.name, delete=False)
            tmp_file.close()
            test.out_file = tmp_file.name

        if cached_goal is not None and cached_goal.out_file is None:
            # the result was cached, we need to compute out_file first
            if cached_goal.res == 'fail' or cached_goal.res == 'sat' or cached_goal.res == 'unsat':
                test.res = cached_goal.res
                test.rlimit = -1
                return
            
            tmp_file = tempfile.NamedTemporaryFile(
                dir=os.path.dirname(test.out_file), suffix='.xyz', delete=False)
            tmp_file.close()

            parent_t = None
            num_tries = 0

            # depending on the server load this computation can sometimes timeout
            while num_tries < 3:
                num_tries += 1
                parent_t = utils_test.IncrementalBenchmarkTest(
                    file=test.file,
                    strat=cached_goal.strat,
                    timeout=num_tries * 3 * test.timeout,
                )
                parent_t.run(out_file=tmp_file.name)

                if parent_t.out_hash == cached_goal.out_hash:
                    break

            cached_goal.out_file = tmp_file.name
            assert parent_t.out_hash == cached_goal.out_hash, 'parent_t.out_hash: %s, cached_goal.out_hash: %s, strat: %s, test: %s, cached_goal: %s, parent_t: %s' % (parent_t.out_hash, cached_goal.out_hash, cached_goal.strat, str(test), str(cached_goal), str(parent_t))

    def evaluate_sequential(self, tests):
        """ Given a list of tests, performs sequential evaluation. """
        self.log.debug('Evaluating %d tests sequentially' % (len(tests)))
        start = time.time()
        num_cached = 0
        num_evaluated = 0

        for test in tests:
            self.assign_cached_result(test)

            if test.res is None:
                self.compute_cached_outputs(test)
                if test.res != 'fail':
                    self.log.debug('Not cached, running: ' + str(test))
                    test.run()
                self.add_test(test)
                num_evaluated += 1
            else:
                num_cached += 1

        self.log.debug('Num cached %d/%d' % (num_cached, len(tests)))
        end = time.time()
        self.log.debug("Done in %s s." % (str(end - start)))
        return num_evaluated

    def evaluate_parallel(self, tests):
        """ Given a list of tests, performs parallel evaluation. """
        self.log.debug('Evaluating %d tests in parallel using %d threads' % (len(tests), self.num_threads))
        start = time.time()
        unscored_tests = []
        for test in tests:
            self.assign_cached_result(test)
            if test.res is None:
                self.compute_cached_outputs(test)
                if test.res != 'fail':
                    unscored_tests.append(test)

        end = time.time()
        self.log.debug("Done in %s s. Found %d/%d results in the cache" %
                     (str(end - start), len(tests) - len(unscored_tests), len(tests)))
        # nothing to do, cache contained all the tests
        if not unscored_tests:
            return 0

        out_file = os.path.join(self.tmp_dir, 'results_%d' % (os.getpid()))
        if not os.path.isdir(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))

        self.log.debug('Running tasks in parallel...')
        start = time.time()
        run_ordered_tasks_and_merge_outputs(unscored_tests, exec_test, get_merge_files_callback(out_file), self.num_threads, self.tmp_dir)
        end = time.time()
        self.log.debug('Done in %s s' % (str(end - start)))

        self.log.debug('Collecting results and updating cache ...')
        start = time.time()
        results = self.load_results(out_file)
        assert len(unscored_tests) == len(results), "%d vs %d" % (len(unscored_tests), len(results))            
        
        for test, scored_test in zip(unscored_tests, results):
            assert test.file == scored_test.file
            assert test.strat == scored_test.strat
            test.res = scored_test.res
            test.rlimit = scored_test.rlimit
            test.runtime = scored_test.runtime
            test.kind = scored_test.kind
            if isinstance(scored_test, utils_test.BenchmarkTest):
                self.add_test(test)
                continue
            test.in_hash = scored_test.in_hash
            test.out_hash = scored_test.out_hash
            test.out_file = scored_test.out_file
            test.probes = scored_test.probes
            test.partial_tasks = scored_test.partial_tasks
            test.bow = scored_test.bow
            test.ast = scored_test.ast
            test.vec = scored_test.vec
            self.add_test(test)

        end = time.time()
        self.log.debug('Done in %s s' % (str(end - start)))
        return len(unscored_tests)

    def add_scored_strategy(self, scored_candidate, status):
        if self.out_file is not None:
            self.data.append((scored_candidate, status))

            if len(self.data) > 5000:
                self.save()

    def save(self):
        self.log.debug('out_file: ' + str(self.out_file))
        if self.out_file is None:
            return

        with open(self.out_file, 'a') as f:
            for candidate, c_status in self.data:
                value = {
                    'task': candidate.benchmarks[0].to_json(),
                    'pruned': candidate.pruned,
                    'solved': candidate.all_solved(),
                    'status': str(c_status)
                }
                f.write(json.dumps(value) + '\n')

        self.data = []


class BenchmarkTester(object):


    def __init__(self, cache_dir=None, verbose=True, num_threads=1):
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.num_threads = num_threads
        if self.cache_dir is not None and not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache = {}
        self.tot_cached = 0

    def get_cache_path(self, file):
        return os.path.join(self.cache_dir, os.path.basename(file) + ".cache")

    def print_cache(self):
        print("Cache Containg %d entries" % len(self.cache))
        files_printed = 0
        for file, value in self.cache.items():
            print(file)
            strategies_printed = 0
            for strategy, test in value.items():
                print('\t%s: %s' % (strategy, str(test)))
                strategies_printed += 1
                if strategies_printed > 3:
                    break

            if len(value) > strategies_printed:
                print('\t%d More Entries...' % (len(value) - strategies_printed))

            files_printed += 1
            if files_printed > 3:
                break

        if len(self.cache) > files_printed:
            print('%d More Files...' % (len(self.cache) - files_printed))

    def size(self):
        res = 0
        for file, value in self.cache.items():
            res += len(value)
        return res

    def save_cache(self):
        if self.cache_dir is None:
            return
        print("Saving Cache Containg %d smt_files with %d strategies in total" % (len(self.cache), self.size()))
        for file, value in self.cache.items():
            ## JSON
            # raw_value = {key: value.ToJSON() for key, value in value.items()}
            # print(raw_value)
            # print(value)
            # with open(self.get_cache_path(file), "w") as f:
            #     f.write(json.dumps(raw_value))

            ## Pickle
            with open(self.get_cache_path(file), "wb") as f:
                pickle.dump(value, f)

    def LoadCacheForTest(self, test):
        if self.cache_dir is None:
            self.cache[test.file] = {}
            return

        if not os.path.isfile(self.get_cache_path(test.file)):
            self.cache[test.file] = {}
            return

        if self.verbose:
            print("Loading Cache for %s" % (self.get_cache_path(test.file)))
            ## JSON
            # with open(self.get_cache_path(test.file), "r") as f:
            # raw_value = json.load(f)
            # print(raw_value)
            # self.cache[test.file] = {key: BenchmarkTest.FromJSON(value) for key, value in raw_value.items()}
            # print(self.cache[test.file])

        ## Pickle
        with open(self.get_cache_path(test.file), "rb") as f:
            self.cache[test.file] = pickle.load(f)

    def add_test(self, test):
        if test.file not in self.cache:
            self.LoadCacheForTest(test)

        cached_test = self.cache[test.file].get(test.strat)
        if cached_test is None or cached_test.timeout < test.timeout:
            self.cache[test.file][test.strat] = test

    def assign_cached_result(self, test):
        # print('assign_cached_result: ' + str(test))
        if test.file not in self.cache:
            self.LoadCacheForTest(test)

        cached_test = self.cache[test.file].get(test.strat)
        if cached_test is None:
            return

        # if the result is sat/unsat then we return the result directly
        # if the cached timeout is higher and the result is unknown then lower timeout does not change anything
        if cached_test.res == "sat" or cached_test.res == "unsat" or cached_test.timeout >= test.timeout:
            test.res = cached_test.res
            test.rlimit = cached_test.rlimit
            test.runtime = cached_test.runtime
            test.probes = cached_test.probes
            # print('assign_cached_result: ' + str(test))
            return

        return

    def load_results(self, file):
        results = []
        with open(file, "r") as f:
            for line in f:
                results.append(utils_test.BenchmarkTest.from_json(json.loads(line)))
        return results

    def evaluate_sequential(self, tests):
        all_cached = True
        for test in tests:
            self.assign_cached_result(test)
            if test.res is None:
                # print("Running: %s" % (test))
                test.Run()
                self.add_test(test)
                all_cached = False

        if not all_cached:
            self.save_cache()

    def evaluate_parallel(self, tests):
        if self.verbose:
            print("Searching for Cached results...")
        start = time.time()
        unscored_tests = []
        for test in tests:
            self.assign_cached_result(test)
            if test.res is None:
                unscored_tests.append(test)

        end = time.time()
        if self.verbose:
            print("Done in %s s. Found %d/%d results in the cache" % (
            str(end - start), len(tests) - len(unscored_tests), len(tests)))
        # nothing to do, cache contained all the tests
        if not unscored_tests:
            return

        out_file = os.path.join(self.tmp_dir, 'results_%d' % (os.getpid()))
        if not os.path.isdir(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))

        if self.verbose:
            print('Running Tasks in Parallel...')
        start = time.time()
        run_ordered_tasks_and_merge_outputs(unscored_tests, exec_test, get_merge_files_callback(out_file), self.num_threads, self.tmp_dir)
        end = time.time()
        if self.verbose:
            print('Done in %s s' % (str(end - start)))
        results = self.load_results(out_file)
        assert len(unscored_tests) == len(results)
        for test, scored_test in zip(unscored_tests, results):
            assert test.file == scored_test.file
            assert test.strat == scored_test.strat
            test.res = scored_test.res
            test.rlimit = scored_test.rlimit
            test.runtime = scored_test.runtime
            test.probes = scored_test.probes
            self.add_test(test)

        self.save_cache()

