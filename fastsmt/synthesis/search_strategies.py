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

import itertools
import heapq
import numpy as np
from abc import ABCMeta, abstractmethod
from fastsmt.utils.test import *
from fastsmt.utils.tester import *
from fastsmt.synthesis.value_counter import *
from fastsmt.utils.strategy import StrategyEnumerator, Strategy
from enum import Enum


class SearchStrategy:
    """ Base class which is wrapper around the search procedure."""
    __metaclass__ = ABCMeta

    def __init__(self, tester):
        self.candidates = []
        self.tester = tester
        self.goal_test = True
        self.num_evaluated = 0
        self.log = logging.getLogger('SearchStrategy')

    def add_counter_example(self, smt_instance):
        self.smt_instances.append(smt_instance)

    @abstractmethod
    def init_population(self, tot_samples, initial_candidate):
        """ Initialize population of strategies.

        :param tot_samples: number of samples to initialize the population with
        :param initial_candidate: initial candidate strategy
        """
        pass

    @abstractmethod
    def extend_population(self, desired_size):
        """ Extend the population with number of new strateies.

        :param desired_size: Number of new strategies to extend the population with
        """
        pass

    @abstractmethod
    def finished_evaluating(self, scored_candidates):
        """ Callback method after evaluation of candidate strategies is finished.

        :param scored_candidates: Scored candidate strategies whose evaluation was finished.
        """
        pass

    def evaluate(self, timeout):
        """ Evaluates all candidate strategies on all smt instances. """
        unscored_tasks = []
        unscored_candidates = []

        for smt_instance, candidate in self.candidates:
            task = BenchmarkGoalTest(
                smt_instance,
                tmp_dir=self.tester.cache_dir.name if self.tester.cache_dir is not None else None,
                strat=candidate.t,
                timeout=timeout,
                tester=self.tester,
            )
            if not candidate.contains_benchmark(task):
                candidate.add_benchmark_test(task)
                unscored_tasks.append(task)
                unscored_candidates.append(candidate)
                
        t1 = time.time()
        if len(unscored_tasks) == 1:
            self.num_evaluated += self.tester.evaluate_sequential(unscored_tasks)
        else:
            self.num_evaluated += self.tester.evaluate_parallel(unscored_tasks)
        t2 = time.time()
        self.log.info('Evaluation finished in ' + str(t2-t1) + ' seconds!')

        for smt_instance, candidate in self.candidates:
            candidate.finished_scoring()
            self.log.debug('Finished scoring: ' + str(candidate.benchmarks[0]))

        self.last_candidates = self.candidates
        self.finished_evaluating(unscored_candidates)

    def get_candidates(self, smt_instance):
        return [candidate for _, candidate in self.last_candidates
                if candidate.benchmarks[0].file == smt_instance]

    def print(self):
        print('Population with %d candidates' % (len(self.candidates)))
        for candidate in self.candidates:
            print('\t' + str(candidate))

            
class ScoredCandidateStatus(Enum):
    SOLVED = 0,
    PRUNED = 1,
    TIMEOUT = 2,
    SUCCESS = 3,
    REDUNDANT = 4


class FormulaEqClass:


    def __init__(self, candidate):
        self.candidate = candidate
        self.count = 1


class ModelSearch(SearchStrategy):
    """ Wrapper around strategy search procedure guided by model. """

    def __init__(self, tester, config, main_model, explore_model):
        """ Initializes the search procedure.

        :param tester: BenchmarkGoalTester to be used for evaluation
        :param config: configuration of the model
        :param main_model: main model to use during the search
        :param explore_model: exploration model to use during the search
        """
        assert isinstance(tester, BenchmarkGoalTester)
        super(ModelSearch, self).__init__(tester)

        self.config = config
        self.main_model = main_model
        self.explore_model = explore_model
        self.explore_rate = self.config['exploration']['init_explore_rate']
        self.min_explore_rate = self.config['exploration']['min_explore_rate']
        self.explore_decay = self.config['exploration']['explore_decay']
        self.enabled_explore = self.config['exploration']['enabled']
        self.next_explore = self.enabled_explore

        if "tactics_config" in config:
            tactics_config = config["tactics_config"]
        else:
            tactics_config = {}

        self.strategy_enum = StrategyEnumerator(**tactics_config)

    def restart(self, smt_instances, valid):
        """ Restarts the search.

        :param smt_instances: new SMT formula instances to be solved
        :param valid: flag whether this is validation run
        """
        self.smt_instances = smt_instances
        self.main_model.smt_instances = smt_instances
        self.main_model.valid = valid
        self.explore_model.smt_instances = smt_instances
        self.explore_model.valid = valid
        self.per_formula_strategies = {None: FormulaEqClass(None)}
        self.clear()
        self.valid = valid

    def clear(self):
        """ Clear the data from previous run. """
        self.valid = False
        self.skip_tests = None

        # all strategies that might be scored in the future
        self.unscored_strategies = {smt_instance: [] for smt_instance in self.smt_instances}
        self.scored_strategies = {smt_instance: [] for smt_instance in self.smt_instances}
        self.best_strategy = {smt_instance: None for smt_instance in self.smt_instances}

        # selected strategies to be evaluated at current step
        self.candidates = []

        # unique sequence count
        self.counter = itertools.count()

        self.stats = {
            'num_pruned': 0,
            'num_redundant': 0,
            'num_success': 0,
            'num_solved': 0,
            'num_timeout': 0,
        }

    def get_stats(self):
        return str(self.stats) + ', scored strategies: %d' % (len(self.scored_strategies))

    def select_candidates(self, desired_size, smt_instance):
        """ Selects best candidate strategies that should be tested.

        :param desired_size: number of candidates to select
        :param smt_instance: smt instance for which synthesis is performed
        :return: number of candidates selected
        """
        model = self.explore_model if self.next_explore or not self.main_model.can_predict() \
                else self.main_model

        for i, entry in enumerate(self.unscored_strategies[smt_instance]):
            candidate_tactics, strategy = entry

            if strategy is None:
                parent = Strategy(self.skip_tests[smt_instance].strat)
                parent.add_benchmark_test(self.skip_tests[smt_instance])
            else:
                parent = strategy

            child_score = model.score_strategy(strategy=candidate_tactics, parent=parent)
            heapq.heappush(self.scored_strategies[smt_instance],
                           (-1 * child_score, next(self.counter), entry))

        self.unscored_strategies[smt_instance] = []

        candidates_added = 0
        while candidates_added < desired_size and self.scored_strategies[smt_instance]:
            priority, count, entry = heapq.heappop(self.scored_strategies[smt_instance])
            new_strategy = Strategy(make_strategy(entry[0]))
            new_strategy.score = -priority
            self.candidates.append((smt_instance, new_strategy))
            candidates_added += 1

        return candidates_added

    def init_population(self, desired_size, initial_candidate=None, smt_instances=None):
        """ Initializes population of strategies.

        :param desired_size: number of strategies to initialize with
        :param initial_candidate: initial candidate strategy, empty if None
        :param smt_instances: list of smt instances for which synthesis should be performed
        """
        TMP_DIR = os.path.join(self.tester.tmp_dir, 'search')
        if TMP_DIR is not None and not os.path.isdir(TMP_DIR):
            os.makedirs(TMP_DIR)
        if smt_instances is None:
            smt_instances = self.smt_instances

        if self.skip_tests is None:
            self.skip_tests = {
                smt_instance:
                    BenchmarkGoalTest(
                        smt_instance,
                        make_strategy(['skip']),
                        TMP_DIR,
                        timeout=10,
                        tester=self.tester)
                for smt_instance in self.smt_instances
            }
            self.tester.evaluate_parallel(self.skip_tests.values())

        self.main_model.reset()
        self.explore_model.reset()

        for smt_instance in smt_instances:
            self.add_candidates_from_strategy(initial_candidate, smt_instance)
            self.select_candidates(desired_size, smt_instance)

    def add_candidates_from_strategy(self, strategy, smt_instance):
        """ Given strategy, expands it with possible tactics and arguments and adds
        all resulting strategies to the list of unscored strategies.

        :param strategy: strategy which should be expanded
        :param smt_instance: smt instance for which synthesis is performed
        :return:
        """
        tactics = [] if strategy is None else get_tactics(strategy.t)
        for tactic in self.strategy_enum.base_tactics:
            if strategy is None:
                parent = Strategy(self.skip_tests[smt_instance].strat)
                parent.add_benchmark_test(self.skip_tests[smt_instance])
            else:
                parent = strategy

            if self.next_explore or not self.main_model.can_predict():
                model = self.explore_model
            else:
                model = self.main_model
            args = model.predict_arguments(tactic.s, parent)
            with_tactic = self.strategy_enum.get_tactic_with_args(tactic.s, args)

            candidate_tactics = tactics + [with_tactic]

            if StrategyEnumerator.is_valid_strategy(candidate_tactics):
                self.unscored_strategies[smt_instance].append(
                    (candidate_tactics,
                     strategy if strategy and strategy.benchmarks else None))

    def extend_population(self, desired_size, smt_instance):
        """ Extends population of strategies with desired number of new strategies.

        :param desired_size: number of strategies to add to the population
        :param smt_instance: smt instance for which synthesis is performed
        """
        return self.select_candidates(desired_size, smt_instance) > 0

    def prune(self, smt_instance):
        """ Prunes all strategies which can not possibly lead to optimal strategy for solving smt instance. """
        new_candidates = []
        for strategy, parent in self.unscored_strategies[smt_instance]:
            if parent and parent.pruned:
                continue
            if parent and (self.best_strategy[smt_instance] is not None) and self.best_strategy[smt_instance].rlimit <= parent.rlimit:
                continue
            new_candidates.append((strategy, parent))
        self.unscored_strategies[smt_instance] = new_candidates

    def add_scored_strategy(self, candidate, status):
        """ Reports outcome of evaluating candidate strategy on smt instance to the model. """
        if not self.valid:
            self.main_model.add_scored_strategy(candidate, status)
        self.tester.add_scored_strategy(candidate, status)

    def finished_evaluating(self, scored_candidates):
        """ Callback function called after evaluation of candidates is finished.

        :param scored_candidates: candidate strategies which were evaluated on smt instances
        """
        if not self.valid:
            self.explore_rate *= self.explore_decay
            self.explore_rate = np.clip(self.explore_rate, self.min_explore_rate, 1)

        for candidate in scored_candidates:
            smt_instance = candidate.benchmarks[0].file

            if candidate.all_solved() and (self.best_strategy[smt_instance] is None or candidate.rlimit < self.best_strategy[smt_instance].rlimit):
                self.best_strategy[smt_instance] = candidate
                self.stats['num_solved'] += 1
                self.add_scored_strategy(candidate, ScoredCandidateStatus.SOLVED)
                self.prune(smt_instance)
                continue

            in_hashes, out_hashes = candidate.get_goal_hashes()
            if in_hashes == out_hashes and (len(self.unscored_strategies[smt_instance]) > 0 or candidate.rlimit < 0 or candidate.failed_benchmarks()):
                self.stats['num_pruned'] += 1
                candidate.pruned = True
                self.add_scored_strategy(candidate, ScoredCandidateStatus.PRUNED)
                continue
            if self.best_strategy[smt_instance] is not None and self.best_strategy[smt_instance].rlimit <= candidate.rlimit:
                self.stats['num_timeout'] += 1
                self.add_scored_strategy(candidate, ScoredCandidateStatus.TIMEOUT)
                continue

            if str(out_hashes) not in self.per_formula_strategies:
                self.per_formula_strategies[str(out_hashes)] = FormulaEqClass(candidate)
                self.add_candidates_from_strategy(candidate, smt_instance)
                self.stats['num_success'] += 1
                self.add_scored_strategy(candidate, ScoredCandidateStatus.SUCCESS)
            else:
                formula = self.per_formula_strategies[str(out_hashes)]
                formula.count += 1

                if formula.candidate.rlimit > candidate.rlimit:
                    formula.candidate.pruned = True
                    formula.candidate = candidate
                    self.add_candidates_from_strategy(candidate, smt_instance)
                    self.stats['num_success'] += 1
                    self.add_scored_strategy(candidate, ScoredCandidateStatus.SUCCESS)
                    self.prune(smt_instance)
                else:
                    if len(self.unscored_strategies[smt_instance]) > 0 or candidate.rlimit < 0 or candidate.failed_benchmarks():
                        self.stats['num_redundant'] += 1
                        candidate.pruned = True
                        self.add_scored_strategy(candidate, ScoredCandidateStatus.REDUNDANT)
                    else:
                        self.add_candidates_from_strategy(candidate, smt_instance)
                        self.stats['num_success'] += 1
                        self.add_scored_strategy(candidate, ScoredCandidateStatus.SUCCESS)
                        self.prune(smt_instance)

        self.candidates = []
        if self.enabled_explore:
            self.log.info('Exploration rate: ' + str(self.explore_rate))
        self.next_explore = (not self.valid and np.random.random() < self.explore_rate and self.enabled_explore)

