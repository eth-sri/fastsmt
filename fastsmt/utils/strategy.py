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

from collections import OrderedDict
from enum import Enum
from fastsmt.language.objects import *

MAX_INT_VALUE = 1000

class StrategyEnumerator:
    """ Class which is wrapper over all possible strategies. It can be queried for
    validity of strategy, random strategy, etc. """

    def __init__(self, all_tactics, allowed_params):
        """ Initializes object of type StrategyEnumerator.

        :param all_tactics: list of strings representing names of tactics
        :param allowed_params: dict which maps tactic name to list of parameters
        """
        self.all_tactics = all_tactics
        self.allowed_params = allowed_params
        self.base_tactics = [Tactic(tactic) for tactic in self.all_tactics]

        self.param_min = OrderedDict()
        self.param_max = OrderedDict()

        for tactic in self.all_tactics:
            self.param_min[tactic] = OrderedDict()
            self.param_max[tactic] = OrderedDict()
            if tactic in self.allowed_params:
                if 'boolean' in self.allowed_params[tactic]:
                    for bool_param in self.allowed_params[tactic]['boolean']:
                        self.param_min[tactic][bool_param] = 0
                        self.param_max[tactic][bool_param] = 1
                if 'integer' in self.allowed_params[tactic]:
                    for int_param, min_value, max_value in self.allowed_params[tactic]['integer']:
                        self.param_min[tactic][int_param] = min_value
                        self.param_max[tactic][int_param] = max_value

    def get_params_per_tactic(self):
        """ Returns number of parameters that need to be supplied for each tactic. """
        return {tactic: len(self.param_max[tactic]) for tactic in self.all_tactics}

    def get_tactic_with_args(self, tactic, args):
        """ Returns With tactic constructed by instantiating tactic with given arguments.
        Arguments should be mapping from tactic name to real number in the interval [0, 1].
        Argument for tactic is scaled to interval [0, max] and rounded to the nearest integer.

        :param tactic: tactic to instantiate
        :param args: arguments to use
        :return: constructed With tactic
        """
        if tactic not in self.allowed_params:
            return Tactic(tactic)

        params = OrderedDict()
        
        for arg, value in args.items():
            true_value = self.param_min[tactic][arg] + value * (self.param_max[tactic][arg] - self.param_min[tactic][arg])
            if (self.allowed_params[tactic].get('boolean') is not None) and (arg in self.allowed_params[tactic]['boolean']):
                params[arg] = False if true_value < 0.5 else True
            else:
                params[arg] = int(true_value)
                
        with_tactic = With(tactic, params)
        return with_tactic

    def extract_params(self, tactics):
        """ Given list of tactics applied so far, extracts parameters associated with each of the tactics."""
        ret = []
        for tactic in tactics:
            if isinstance(tactic, With):
                tmp_params = OrderedDict()
                for param_name, param_value in tactic.params.items():
                    if isinstance(param_value, bool):
                        assert self.param_max[tactic.s][param_name] == 1
                        value = int(param_value)
                        tmp_params[param_name] = float(value)
                    else:
                        value = int(param_value)
                        tmp_params[param_name] = float(value) / self.param_max[tactic.s][param_name]
                ret.append(tmp_params)
            else:
                ret.append([])
        return ret

    @staticmethod
    def is_valid_strategy(tactics):
        """ Checks whether list of tactics represents a valid strategy.
        Strategy is valid if it:
          - starts with "skip"
          - does not contain "skip" at any other position
          - does not contain two successive applications of the same tactic
        """
        if tactics[0].s != 'simplify':
            return False

        for i in range(1, len(tactics)):
            if tactics[i - 1].s == tactics[i].s:
                return False

        return True


class Strategy:
    """ Object which represents an application of strategy to a set of benchmarks. """

    def __init__(self, t):
        """ Initializes an object of type Strategy.

        :param t: tactics which represent the strategy.
        """
        self.t = t
        self.benchmarks = []
        self.rlimit = None
        self.runtime = None
        self.mask = None
        self.pruned = False
        self.only_smt_instance = None
        self.score = 0

    def to_json(self):
        """ Converts strategy object to JSON format. """
        res = {}
        for key, value in self.__dict__.items():
                res[key] = value
        return res

    def to_fast_text_solved_pruned(self):
        return '__label__Solved' + str(self.all_solved()) + ' ' + '__label__Pruned' + str(self.pruned) + ' '.join(
            [str(t) for t in get_tactics(self.t)])

    def to_fast_text_solved(self):
        return '__label__' + str(self.all_solved()) + ' ' + ' '.join([str(t) for t in get_tactics(self.t)])

    def to_fast_text_pruned(self):
        return '__label__' + str(self.pruned) + ' ' + ' '.join([str(t) for t in get_tactics(self.t)])

    def to_fast_text_pruned_features(self):
        return '__label__' + str(self.pruned) + ' ' + ' '.join(self.benchmarks[0].features)

    def contains_benchmark(self, task):
        for benchmark in self.benchmarks:
            if benchmark.file == task.file:
                return True
        return False

    def get_probes(self):
        probes = []
        for benchmark in self.benchmarks:
            probes.append(benchmark.probes)
        return probes

    def get_bow(self):
        bow = []
        for benchmark in self.benchmarks:
            bow.append(benchmark.bow)
        return bow

    def get_ast(self):
        ast = []
        for benchmark in self.benchmarks:
            ast.append(benchmark.ast)
        return ast

    def get_vec(self):
        vec = []
        for benchmark in self.benchmarks:
            vec.append(benchmark.vec)
        return vec

    def get_goal_hashes(self):
        in_hashes = []
        out_hashes = []
        for benchmark in self.benchmarks:
            in_hashes.append(benchmark.in_hash)
            out_hashes.append(benchmark.out_hash)
        return (in_hashes, out_hashes)

    def failed_benchmarks(self):
        for benchmark in self.benchmarks:
            if benchmark.res == 'fail':
                return True
        return False

    def add_benchmark_test(self, task):
        assert not self.contains_benchmark(task)
        self.benchmarks.append(task)

    def cost(self):
        return (-self.mask.count('1'), self.rlimit)

    def all_solved(self):
        return self.rlimit > 0 and self.runtime > 0 and self.mask.count('1') == len(self.mask)

    def finished_scoring(self):
        self.rlimit = 0
        self.runtime = 0
        self.mask = ''

        for benchmark in self.benchmarks:
            self.rlimit += benchmark.rlimit
            self.runtime += benchmark.runtime

            if benchmark.is_solved() and (benchmark.rlimit <= 0 or benchmark.runtime <= 0):
                # print(benchmark)
                benchmark.res = 'fail'
                # assert False, 'rlimit is -1 for sat/unsat goal'

            self.mask += '1' if (benchmark.res == 'sat' or benchmark.res == 'unsat') else '0'

    def __str__(self):
        return "%s" % (self.benchmarks[0].nice_str())
        # return "rlimit(%s), runtime(%s), mask(%s), strategy(%s)" % (str(self.rlimit), str(self.runtime), str(self.mask), str(self.t))

    def __repr__(self):
        return str(self)


class StrategyType(Enum):
    RANDOM = 0,
    ENUMERATIVE = 1,
