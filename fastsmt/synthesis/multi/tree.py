from collections import OrderedDict
from fastsmt.language.objects import *
from fastsmt.synthesis.multi.predicates import DefaultClassifierModel, ComparisonPredicate
from fastsmt.utils.test import BenchmarkGoalTest
from fastsmt.utils.strategy import Strategy
from fastsmt.synthesis.search_strategies import ScoredCandidateStatus

class StrategyTree:
    """ Class which represents the tree constructed from a set of strategies. """

    def __init__(self, tester, classifier=None, ground_truth_classifier=None, prefix=None, timeout=10):
        """ Initializes StrategyTree object.

        :param tester: object of type BenchmarkGoalTester used to run the tests
        :param classifier: classifier used to make decisions
        :param ground_truth_classifier: classifier which holds the ground truth information
        :param prefix: prefix of the strategy to which root of the tree corresponds to
        """
        self.tree = OrderedDict()
        self.classifier = classifier

        # used during training (not available during inference)
        self.gt_classifier = ground_truth_classifier
        self.num_strategies = 0
        self.tester = tester
        self.prefix = prefix if prefix is not None else []
        self.timeout = timeout

    def get_classifier_positions(self):
        """ Returns list of all the positions in the tree where a decision has to be made
        (i.e., we need to select between multiple strategies).
        """
        res = [self] if self.classifier else []
        for key, value in self.tree.items():
            res += value.get_classifier_positions()
        return res

    def reset_stats(self):
        """ Resets all statistics in the classifier. """
        for classifier in self.get_classifier_positions():
            classifier.classifier.reset_stats()

    def add(self, strategy):
        """ Inserts given strategy into the tree. """
        self.num_strategies += 1
        tactic, strategy_suffix = head_strategy(strategy)

        if str(tactic) not in self.tree:
            self.tree[str(tactic)] = StrategyTree(
                tester=self.tester,
                classifier=None,
                ground_truth_classifier=self.gt_classifier,
                prefix=self.prefix + [tactic],
                timeout=self.timeout)
        if strategy_suffix:
            self.tree[str(tactic)].add(strategy_suffix)

        if self.classifier is None and self.num_strategies > 1:
            self.classifier = DefaultClassifierModel(str(tactic), parent=self)

    def get_test(self, smt_instance, tactics):
        """ Evaluates strategy on a smt instance. """
        test = BenchmarkGoalTest(
            file=smt_instance,
            strat=make_strategy(tactics),
            tmp_dir=None,
            timeout=self.timeout,
            tester=self.tester)
        self.tester.evaluate_sequential([test])
        strat = Strategy(test.strat)
        strat.benchmarks.append(test)
        strat.finished_scoring()
        self.tester.add_scored_strategy(strat, ScoredCandidateStatus.SUCCESS)
        return test

    def classify(self, smt_instance):
        """ Compute which strategy to apply for a given smt instance. """
        try:
            return make_strategy(self.classify_inner(smt_instance, [from_string('skip')]))
        except ValueError as error:
            print("Error", error)
            return None

    def classify_inner(self, smt_instance, tactics):
        """ Given already applied tactics, computes which tactic to apply next for a given smt instance. """
        if len(self.tree) == 0:
            return []

        test = self.get_test(smt_instance, tactics)

        if len(self.tree) == 1:
            tactic = list(self.tree.keys())[0]
        else:
            best_tactics = None
            if self.gt_classifier:
                best_tactics = self.gt_classifier.classify(
                    test,
                    self.tree.keys(),
                    synthesized_tactics=tactics)
            tactic = self.classifier.classify(test, self.tree.keys(), best_tactics=best_tactics)

        tactics.append(from_string(tactic))
        return [from_string(tactic)] + self.tree[tactic].classify_inner(smt_instance, tactics)

    def print(self, depth=0):
        """ Prints tree in format with indentations. """
        indent = '  ' * depth
        if self.classifier:
            print(indent, 'Classifier: ' + str(self.classifier), self.classifier.stats())
        print(indent, '#Strategies:', self.num_strategies)
        for key, value in self.tree.items():
            print(indent, key)
            value.print(depth + 1)

    def get_smt2(self):
        if len(self.tree) == 0:
            return ''
        if len(self.tree) == 1:
            tactic = list(self.tree.keys())[0]
            ret = from_string(tactic).to_smt2()
            if len(self.tree[tactic].tree) > 0:
                ret += ' ' + self.tree[tactic].get_smt2()
            return ret
        return self.classifier.to_smt2(self.tree)





