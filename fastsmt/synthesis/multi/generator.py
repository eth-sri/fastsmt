import itertools
import logging
import math
import random
from enum import Enum
from fastsmt.language.objects import *
from fastsmt.synthesis.multi.predicates import *
from fastsmt.synthesis.search_strategies import ScoredCandidateStatus
from fastsmt.utils.strategy import Strategy
from fastsmt.utils.constants import TIMEOUT
from fastsmt.utils.test import BenchmarkGoalTest

class ProbeType(Enum):
    Default = 1
    Predicate = 2
    NN = 3
    TryFor = 4
    Predicate_TryFor = 5
    SMT = 6

class ProbesGenerator:
    """ Class responsible for generating all possible classifiers. """

    def __init__(self, strategies, smt_instances, tester, prefix_scorer_train, max_timeout):
        self.probe_names_int = ['num-consts', 'num-exprs', 'size']
        self.probe_names_bool = ['is-qfbv-eq', 'is-unbounded', 'is-pb']

        self.per_tactic_runtimes = {}
        self.probe_values = {probe_name: [] for probe_name in self.probe_names_int}
        self.log = logging.getLogger('ProbesGenerator')
        self.max_timeout = max_timeout

        for strategy in strategies:
            self.initialize(strategy, smt_instances, tester)

        self.prefix_scorer_train = prefix_scorer_train

    def initialize_division_values(self):
        """ For each pair of probes (p_1, p_2) creates a probe p_1 / p_2. """
        for numerator, denumerator in itertools.product(self.probe_names_int, self.probe_names_int):
            if numerator == denumerator:
                continue

            values = []
            for n, d in zip(self.probe_values[numerator], self.probe_values[denumerator]):
                value = 0 if d == 0 else n / d
                values.append(value)
            self.probe_values['%s/%s' % (numerator, denumerator)] = values

    def initialize(self, strategy, smt_instances, tester):
        """
        Randomly samples from formulas for each strategy to initialize statistics.
        This will, for example, be used to determine runtimes in TryFor.

        :param strategy: strategy for which statistics should be calculated
        :param smt_instances: list of smt instances
        :param tester: object of type BenchmarkGoalTester
        """
        self.log.info('Initializing generator for strategy %s!' % str(strategy))

        strategy_tactics = get_tactics(strategy)

        assert len(smt_instances) > 0
        instances = random.sample(smt_instances, min(len(smt_instances), 10))

        tactics = []
        prev_runtime = {}
        for i, tactic in enumerate(strategy_tactics):
            tactics.append(tactic)
            tests = []

            for smt_instance in instances:
                test = BenchmarkGoalTest(
                           file=smt_instance,
                           strat=make_strategy(tactics),
                           tmp_dir=None,
                           timeout=self.max_timeout,
                           tester=tester)
                tests.append(test)

            tester.evaluate_parallel(tests)

            for test in tests:
                if test.res == 'fail':
                    continue
                # update cache
                strat = Strategy(test.strat)
                strat.benchmarks.append(test)
                strat.finished_scoring()
                tester.add_scored_strategy(strat, ScoredCandidateStatus.SUCCESS)
                self.add(test, None)

                if i == 0:
                    prev_runtime[smt_instance] = 0
                test_runtime = max(0.0, test.runtime - prev_runtime[smt_instance])
                prev_runtime[smt_instance] += test_runtime

                assert test_runtime >= 0, str(test)

                if str(tactic) not in self.per_tactic_runtimes:
                    self.per_tactic_runtimes[str(tactic)] = []
                self.per_tactic_runtimes[str(tactic)].append(test_runtime)
        #tester.save_cache()

    def add(self, test, best_tactic):
        for name in self.probe_names_int:
            self.probe_values[name].append(test.get_probe_value(name))

    def gen_probes(self, values, parent, type, prefix, training_dataset):
        """ Generates all possible probes of the given type.
        :param values: list of tactics that can possibly be applied
        :param parent: parent of the classifier
        :param type: type of probe to synthesize
        :return: probes of given type
        """
        if type == ProbeType.SMT:
            assert False, 'Not supported!'
            # yield self.gen_smt_probe(parent, prefix, training_dataset, values)
        if type == ProbeType.Default:
            for probe in self.gen_default_probes(values, parent):
                yield probe
        if type == ProbeType.Predicate or type == ProbeType.Predicate_TryFor:
            for probe in self.gen_bool_probes(values, parent):
                yield probe

            for probe in self.gen_comparison_probes(values, parent):
                yield probe
        if type == ProbeType.TryFor or type == ProbeType.Predicate_TryFor:
            for probe in self.gen_try_for_probes(values, parent):
                yield probe

    @staticmethod
    def to_float(s):
        while s[-1] == '?':
            s = s[:-1]
        return float(s)

    def gen_default_probes(self, values, old_classifier):
        """
        Generates all classifiers which
        :param values: list of tactics that can possibly be applied
        :param old_classifier: old classifier in whose place is division predicate synthesized
        :return: predicates of type DefaultClassifierModel
        """
        for value in values:
            yield DefaultClassifierModel(value, parent=old_classifier.parent)
    def gen_comparison_probes(self, values, old_classifier):
        """
        Generates all probes which consist of comparison of two other probes.
        :param values: list of tactics that can possibly be applied
        :param old_classifier: old classifier in whose place is division predicate synthesized
        :return: predicates of type ComparisonPredicate
        """
        percentiles = list(range(5, 100, 5))
        for probe_name in self.probe_names_int:
            for percentile in percentiles:
                assert len(self.probe_values[probe_name]) > 0, 'Probe values empty for ' + probe_name
                const = np.percentile(self.probe_values[probe_name], percentile)
                const = int(const + 0.5)

                yield ComparisonPredicate(old_classifier.parent,
                                          DefaultClassifierModel(values[0]),
                                          DefaultClassifierModel(values[0]),
                                          PredicateOp.GT, probe_name, const)

    def gen_bool_probes(self, values, old_classifier):
        """
        Generates all probes which consist of checking whether probe evaluates to true or false.
        :param values: list of tactics that can possibly be applied
        :param old_classifier: old classifier in whose place is division predicate synthesized
        :return: predicates of type ComparisonPredicate
        """
        for probe_name in self.probe_names_bool: #, 'is-qfbv'
            for value_id_true in range(len(values)):
                for value_id_false in range(value_id_true + 1, len(values)):
                    yield BoolPredicate(old_classifier.parent,
                                        DefaultClassifierModel(values[value_id_true]),
                                        DefaultClassifierModel(values[value_id_false]),
                                        probe_name)

                    yield BoolPredicate(old_classifier.parent,
                                        DefaultClassifierModel(values[value_id_true]),
                                        DefaultClassifierModel(values[value_id_false]),
                                        probe_name)

    def gen_try_for_probes(self, values, old_classifier):
        """
        Generates all TryFor predicates.
        :param values: list of tactics that can possibly be applied
        :param old_classifier: old classifier in whose place is division predicate synthesized
        :return: predicates of type TryForPredicate
        """
        percentiles = list(range(5, 100, 5))
        for try_tactic in values:
            tactic_runtimes = self.per_tactic_runtimes.get(try_tactic)
            timeouts = set()
            for percentile in percentiles:
                timeout = 1000 * np.percentile(tactic_runtimes, percentile)
                timeouts.add(int(math.ceil(timeout / 100.0)) * 100 / 1000)

            for timeout in timeouts:
                # print(timeout)
                for default_tactic in values:
                    if try_tactic == default_tactic:
                        continue

                    yield TryForPredicate(old_classifier.parent,
                                          DefaultClassifierModel(default_tactic, parent=old_classifier.parent),
                                          try_tactic, timeout)
