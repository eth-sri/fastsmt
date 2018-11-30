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

import shutil
import tempfile
import time

from fastsmt.language.objects import *
from fastsmt.utils.constants import *
from fastsmt.utils.runners import *
from fastsmt.utils.tokenizer import Tokenizer
import fastsmt.utils as utils

INF = 10**20

#TODO: create dictionary with keys {ast, vec, bow} to allow for more feature types
class BenchmarkGoalTest(object):
    """ Class which represents test that should be evaluated with a strategy consisting of single tactic. """

    def __init__(self, file, strat, tmp_dir, tester=None, timeout=None):
        """Initializes BenchmarkGoalTest.

        :param file: SMT2 file which contains goal
        :param strat: strategy which should be applied to the goal
        :param tmp_dir: temporary directory where cache should be saved
        :param tester: tester object which should communicate with cache
        :param timeout: time limit used for the execution
        """
        self.file = file
        self.strat = str(strat) if strat is not None else None
        self.timeout = timeout if timeout is not None else 60

        self.out_file = None
        if tmp_dir is not None:
            tmp_file = tempfile.NamedTemporaryFile(suffix='.abc',dir=tmp_dir, delete=False)
            tmp_file.close()
            self.out_file = tmp_file.name

        # output
        self.res = None
        self.rlimit = 0
        self.runtime = 0
        self.in_hash = None
        self.out_hash = None
        self.probes = None
        self.bow = None
        self.ast = None
        self.vec = None
        self.partial_tasks = None
        self.kind = 'BenchmarkGoalTest'

        self.cached_goal = tester.find_cached_prefix(self.file, str(self.strat)) if tester is not None else None
        if self.cached_goal is not None:
            if str(self.cached_goal.strat) == str(self.strat):
                self.kind = self.cached_goal.kind
                self.res = self.cached_goal.res
                self.rlimit = self.cached_goal.rlimit
                self.runtime = self.cached_goal.runtime
                self.in_hash = self.cached_goal.in_hash
                self.out_hash = self.cached_goal.out_hash
                self.out_file = self.cached_goal.out_file
                self.probes = self.cached_goal.probes
                self.bow = self.cached_goal.bow
                self.ast = self.cached_goal.ast
                self.vec = self.cached_goal.vec
            else:
                self.rlimit += self.cached_goal.rlimit
                self.runtime += self.cached_goal.runtime

    @staticmethod
    def from_json(entries):
        obj = BenchmarkGoalTest(None, None, None)
        obj.__dict__.update(entries)
        return obj

    def to_json(self):
        res = {}
        for key, value in self.__dict__.items():
            if key != 'tester' and key != 'cached_goal':
                res[key] = value
        return res

    def get_nn_value(self, name, app_model):
        assert name.startswith('NN_')
        assert False, 'Not supported right now'
        # idx = int(name[3:])
        #
        # tactics = list(map(str, get_tactics(from_string(self.strat))))
        # probes = test.probes + test.bow
        #
        # x_tactics, x_probes = app_model.get_featurized(tactics[1:], probes)
        # probs = app_model.nn.predict(np.array(x_tactics), np.array(x_probes))
        #
        # return probs[idx]

    def get_probe_value(self, name):
        if self.res == 'fail':
            return -1

        # a_1*probe_1+a_2*probe_2
        if '+' in name:
            e1, e2 = name.split('+')
            alpha1, probe1 = e1.split('*')
            alpha2, probe2 = e2.split('*')
            alpha1 = int(alpha1)
            alpha2 = int(alpha2)
            value1 = self.probes[PROBE_TO_ID[probe1]]
            value2 = self.probes[PROBE_TO_ID[probe2]]
            return alpha1 * value1 + alpha2 * value2

        assert name in PROBE_TO_ID
        return self.probes[PROBE_TO_ID[name]]

    """
    Runs the test and returns result (sat, unsat or unknown), rlimit count and runtime.
    """
    def run(self):
        if self.res is not None:
            return self.res, self.rlimit, self.runtime

        if self.cached_goal:
            input_file = self.cached_goal.out_file
            unscored_strategy_suffix = get_strategy_suffix(self.strat, self.cached_goal.strat)
            input_strategy = make_strategy(unscored_strategy_suffix)
        else:
            assert len(get_tactics(self.strat)) == 1
            input_file = self.file
            input_strategy = self.strat

        assert input_file

        t = GoalRunnerThread(input_file, self.out_file, input_strategy, self.timeout)
        success, rlimit, res, in_hash, out_hash, runtime, probes = t.Run()

        # if res is not None:
        #     ast_thread = ASTRunner(self.out_file, self.timeout)
        #     ast_compressed = ast_thread.Run()
        #     self.ast = ast_compressed

        tokenizer = Tokenizer()
        formula_idx = tokenizer.tokenize(self.out_file)
        bow = tokenizer.bag_of_words(formula_idx)

        self.bow = bow
        self.vec = []  # vec
            
        self.in_hash = in_hash
        self.out_hash = out_hash

        if (not success): # or (ast_compressed is None):
            self.res = 'fail'
            self.rlimit = -1
            self.runtime = -1
            self.probes = probes
        else:
            # solved from subgoal
            # it's possible that smt solver 'gives up' before the time limit in which case rlimit > 0

            self.res = res

            self.rlimit += rlimit
            self.runtime += runtime
            self.probes = probes

        return self.res, self.rlimit, self.runtime

    def is_solved(self):
        return self.res == 'sat' or self.res == 'unsat'

    def __str__(self):
        return "BenchmarkGoalTest(output=[%s, %s, %s], in_hash(%s), out_hash(%s), input[%s %s %s],probes=[%s],bow=[%s],ast=[%s],vec=[%s])" % (
            str(self.res), str(self.rlimit), str(self.runtime),
            self.in_hash, self.out_hash,
            self.file, str(self.strat), str(self.timeout), str(self.probes), str(self.bow), str(self.ast),
            str(self.vec))

    def nice_str(self):
        tactics = get_tactics(self.strat)
        return '[' + ','.join([tactic.compact_str() for tactic in tactics]) + ']'

    def __repr__(self):
        return str(self)

"""
Wrapper class for test which should be evaluated with a strategy consisting of multiple tactics.
"""
class IncrementalBenchmarkTest(object):

    def __init__(self, file, strat, timeout=None, no_cache=False, out_file=None):
        self.file = file
        self.strat = str(strat)
        self.timeout = timeout if timeout is not None else 60
        self.no_cache = no_cache
        self.out_file = out_file
        self.kind = 'IncrementalBenchmarkTest'

        # output
        self.in_hash = None
        self.out_hash = None
        self.res = None
        self.rlimit = None
        self.runtime = None
        self.out_hash = None
        self.probes = None
        self.bow = None
        self.ast = None
        self.vec = None

        self.partial_tasks = []

    @staticmethod
    def from_json(entries):
        obj = IncrementalBenchmarkTest(None, None)
        obj.__dict__.update(entries)
        return obj

    def to_json(self):
        res = {}
        for key, value in self.__dict__.items():
            res[key] = value
        return res

    def run(self, out_file=None):
        start = time.time()

        tests = []
        tactics = get_tactics(str(self.strat))

        tmp_tester = utils.tester.BenchmarkGoalTester()
        for prefix_length in range(len(tactics)):
            task = BenchmarkGoalTest(
                file=self.file,
                strat=make_strategy(tactics[:prefix_length + 1]),
                tmp_dir=tmp_tester.cache_dir.name,
                timeout=self.timeout,
                tester=tmp_tester,
            )

            tmp_tester.evaluate_sequential([task])

            if self.in_hash is None:
                self.in_hash = task.in_hash
            self.out_hash = task.out_hash
            self.bow = task.bow
            self.ast = task.ast
            self.vec = task.vec

            tests.append(task)
            self.partial_tasks.append(task.to_json())

            if task.res == 'fail':
                self.out_hash = None
                self.res = 'fail'
                self.rlimit = -1
                break
            if task.is_solved():
                break

        tokenizer = Tokenizer()
        formula_idx = tokenizer.tokenize(tests[-1].out_file)
        bow = tokenizer.bag_of_words(formula_idx)

        self.bow = bow
        self.vec = [] #vec

        if out_file is not None:
            assert len(tactics) == len(tests) or tests[-1].IsSolved(), 'Expected to evaluate all tactics: %s' % (
            tests[-1])
            shutil.copyfile(tests[-1].out_file, out_file)

        self.res = tests[-1].res
        self.rlimit = 0
        self.runtime = time.time() - start
        self.rlimit = sum(test.rlimit for test in tests)
        self.out_hash = tests[-1].out_hash
        self.probes = tests[-1].probes

        assert self.partial_tasks is not None
        assert len(self.partial_tasks) > 0

    def is_solved(self):
        return self.res == 'sat' or self.res == 'unsat'

    def __str__(self):
        return "IncrementalBenchmarkTest(output=[%s, %s, %s], input[%s %s %s], out_hash=%s, out_file=%s)" % (
            str(self.res), str(self.rlimit), str(self.runtime),
            self.file, str(self.strat), str(self.timeout), self.out_hash, self.out_file
        )

    def __repr__(self):
        return str(self)


class BenchmarkTest(object):


    def __init__(self, file, strat=None, timeout=None):
        # input
        self.file = file
        self.strat = str(strat) if strat is not None else 'default'
        self.timeout = timeout if timeout is not None else 60
        self.kind = 'BenchmarkTest'

        # output
        self.res = None
        self.rlimit = None
        self.runtime = None
        self.probes = None
        self.out_file = None

    # used to deserialize and serialize tests results into cache
    @staticmethod
    def from_json(entries):
        obj = BenchmarkTest(None)
        obj.__dict__.update(entries)
        return obj

    def to_json(self):
        return self.__dict__

    def run(self):
        if self.res is not None:
            return self.res, self.rlimit, self.runtime

        t = RunnerThread(self.strat, self.file, self.timeout)
        start = time.time()
        res, rlimit = t.Run()
        end = time.time()

        self.res = res
        # it's possible that smt solver 'gives up' before the time limit in which case rlimit > 0
        self.rlimit = rlimit
        self.runtime = end - start
        return self.res, self.rlimit, self.runtime

    def is_solved(self):
        return self.res == 'sat' or self.res == 'unsat'

    def __str__(self):
        return "BenchmarkTest(output=[%s, %s, %s], input[%s %s %s])" % (
            str(self.res), str(self.rlimit), str(self.runtime),
            self.file, str(self.strat), str(self.timeout)
        )

    def __repr__(self):
        return str(self)


