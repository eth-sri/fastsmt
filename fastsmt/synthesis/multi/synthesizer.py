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

import logging
from abc import abstractmethod

import math
import numpy as np

from fastsmt.language.objects import *
from fastsmt.synthesis.multi.generator import ProbesGenerator, ProbeType
from fastsmt.synthesis.multi.predicates import GroundTruthClassifierModel, DefaultClassifierModel, \
    ComparisonPredicate, BoolPredicate
from fastsmt.synthesis.multi.tree import StrategyTree
from fastsmt.synthesis.search.models import TrainedModel
from fastsmt.utils.constants import TIMEOUT
from fastsmt.utils.utilities import evaluate_candidate_strategy

class PrefixBestScorer:
    """ Calculates best strategy for each prefix. """

    def __init__(self, strategies, smt_tester, smt_instances, max_timeout, num_strategies=0, f_lambda=0):
        """ Initializes the PrefixBestScorer object.
         
        :param strategies: list of strategies that should be evaluated 
        :param smt_tester: BenchmarkGoalTester used to evaluated
        :param smt_instances: list of smt instances to be evaluated on
        :param max_timeout: maximum allowed time limit for each run
        :param f_lambda: weight for solving unsolved formula
        """
        self.f_lambda = f_lambda
        self.best_task_per_smt = {}

        eval = np.full((len(strategies), len(smt_instances)), -1)

        for strat_idx, strategy in enumerate(strategies):
            tasks = evaluate_candidate_strategy(smt_tester, strategy, smt_instances, max_timeout)

            for task_idx, task in enumerate(tasks):
                tactics = get_tactics(task.strat)
                for i in range(len(tactics) + 1):
                    prefix = ' '.join(list(map(str, tactics[:i])))
                    key = (task.file, prefix)
                    if key not in self.best_task_per_smt:
                        self.best_task_per_smt[key] = TIMEOUT

                    assert task.rlimit < TIMEOUT, 'Found rlimit greater than TIMEOUT, increate the TIMEOUT!: ' + str(task.rlimit)
                    if task.is_solved():
                        eval[strat_idx, task_idx] = task.rlimit
                    if task.is_solved() and task.rlimit < self.best_task_per_smt[key]:
                        self.best_task_per_smt[key] = task.rlimit

        solved = {}
        strat_used = {}
        self.ord_strat = []

        for it in range(num_strategies):
            best_idx = None
            best_cnt = (-1, -1)
            for strat_idx in range(len(strategies)):
                if strat_idx in strat_used:
                    continue
                cnt = (0, 0)
                for smt_idx in range(len(smt_instances)):
                    if smt_idx not in solved and eval[strat_idx, smt_idx] > 0:
                        cnt = (cnt[0] + self.f_lambda, cnt[1] - self.f_lambda * eval[strat_idx, smt_idx])
                    elif eval[strat_idx, smt_idx] > 0:
                        cnt = (cnt[0] + (1 - self.f_lambda), cnt[1] - (1 - self.f_lambda) * eval[strat_idx, smt_idx])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = strat_idx

            if best_idx is None:
                break

            print('Best idx: ',best_idx,', covers ',best_cnt)
            print(strategies[best_idx])
            for smt_idx in range(len(smt_instances)):
                if eval[best_idx, smt_idx] > 0:
                    solved[smt_idx] = 1
            self.ord_strat.append(strategies[best_idx])
            strat_used[best_idx] = 1

    def get_best_for_prefix(self, task, prefix, tactic):
        """ Returns best rlimit possible to get by appending tactic to prefix. """
        key = (task.file, ' '.join(list(map(str, prefix + [tactic]))))
        assert key in self.best_task_per_smt, key
        return self.best_task_per_smt.get(key)


class ClassifierEval:
    """ Object which handles evaluation of set of strategies on a set of smt formula instances. """

    def __init__(self, strategies, smt_tester, smt_instances, max_timeout):
        """ Initialize object of type ClassifierEval.

        :param strategies: list of strategies to be evaluated
        :param smt_tester: BenchmarkGoalTester which performs evaluation
        :param smt_instances: list of smt instances on which strategies should be evaluated
        :param max_timeout: time limit for running the strategies
        """
        synthesized_model = TrainedModel('Synthesized', None, None, None)
        self.log = logging.getLogger('ClassifierEval')
        
        for strategy in strategies:
            synthesized_model.strategies.append(strategy)

        self.log.info('Starting to solve %d instances with %d strategies' % (len(smt_instances), len(strategies)))
        self.model_task, self.all_tasks = synthesized_model.solve_instances(smt_tester, smt_instances, max_timeout, True)

        self.smt_instances = smt_instances

        self.smt_tester = smt_tester
        self.max_timeout = max_timeout

    def evaluate(self, classifier):
        """ Evaluate classifier - return how many optimal branches are correctly predicted and other stats. """
        correct = 0
        solved = 0
        total = 0
        rlimit = 0
        classifier.reset_stats()
        for i, smt_instance in enumerate(self.smt_instances):
            if not self.model_task[i].is_solved():  # skip instances that cannot be solved
                continue

            total += 1

            strategy = classifier.classify(smt_instance)
            tasks = evaluate_candidate_strategy(self.smt_tester, strategy, [smt_instance], self.max_timeout)
            assert len(tasks) == 1

            if tasks[0].is_solved() and tasks[0].rlimit == self.model_task[i].rlimit:
                assert tasks[0].res == self.model_task[i].res
                correct += 1
            if tasks[0].is_solved():
                solved += 1
                rlimit += tasks[0].rlimit

        return correct, total, solved, rlimit


class MultiProgramSynthesizer:
    """ Class which synthesizes strategy with branches. """

    def __init__(self, strategies, smt_tester, smt_instances_train, smt_instances_valid, max_timeout, leaf_size, num_strategies, f_lambda):
        """ Initializes MultiProgramSynthesizer.

        :param strategies: list of synthesized strategies
        :param smt_tester: tester object to be used to run tasks
        :param smt_instances_train: smt formulas used for training
        :param smt_instances_valid: smt formulas used for validation
        :param max_timeout: time limit for execution
        :param leaf_size: size after which node is considered leaf (only default tactic is synthesized)
        :param f_lambda: weight for solving unsolved formula
        """
        self.leaf_size = leaf_size
        self.num_strategies = num_strategies
        self.tester = smt_tester
        self.log = logging.getLogger('MultiProgramSynthesisOpt')
        self.eval_train = ClassifierEval(strategies, smt_tester, smt_instances_train, max_timeout)
        self.prefix_scorer_train = PrefixBestScorer(strategies, smt_tester, smt_instances_train, max_timeout, num_strategies, f_lambda)
        self.log.info('Eval train constructed')

        strategies = self.prefix_scorer_train.ord_strat
        print('keeping strategies: ',strategies)
        self.eval_train = ClassifierEval(strategies, smt_tester, smt_instances_train, max_timeout)
        self.prefix_scorer_train = PrefixBestScorer(strategies, smt_tester, smt_instances_train, max_timeout)

        smt_instances_valid_train, smt_instances_valid_blind = smt_instances_valid, []
        self.eval_valid = ClassifierEval(strategies, smt_tester, smt_instances_valid_train, max_timeout)
        self.eval_valid_blind = ClassifierEval(strategies, smt_tester, smt_instances_valid_blind, max_timeout)
        self.gt_classifier = GroundTruthClassifierModel()
        self.log.info('Eval valid constructed')

        #smt_tester.save_cache()

        self.prefix_scorer_valid = PrefixBestScorer(strategies, smt_tester, smt_instances_valid, max_timeout)

        self.gt_classifier.add_tasks(self.eval_train.all_tasks)
        self.gt_classifier.add_tasks(self.eval_valid.all_tasks)
        self.gt_classifier.add_tasks(self.eval_valid_blind.all_tasks)

        self.strategy_tree = StrategyTree(self.tester, ground_truth_classifier=self.gt_classifier, timeout=max_timeout)
        for strategy in strategies:
            self.strategy_tree.add(strategy)
        self.strategy_tree.print()

        self.log.info('Multi synthesizer initialized!')
        
        self.probe_gen = ProbesGenerator(
            strategies, smt_instances_train, self.tester, self.prefix_scorer_train, max_timeout)
        #self.tester.save_cache()

    @staticmethod
    def valid_split(smt_instances):
        """ Splits instances in two halves. """
        size = int(len(smt_instances) / 2)
        return smt_instances[:size], smt_instances[size:]

    def close(self):
        self.log.info('Closing the multi-synthesizer!')
        self.strategy_tree.tester.close()

    @staticmethod
    def get_parent(classifier):
        if isinstance(classifier.parent, StrategyTree):
            return classifier.parent
        return MultiProgramSynthesizer.get_parent(classifier.parent)

    @staticmethod
    def get_prefix(classifier):
        return MultiProgramSynthesizer.get_parent(classifier).prefix

    @staticmethod
    def replace_classifier(old_classifier, new_classifier):
        """ Replaces old classifier in the tree with new classifier. """
        assert new_classifier.parent == old_classifier.parent

        if isinstance(old_classifier.parent, StrategyTree):
            old_classifier.parent.classifier = new_classifier
        else:
            #assert isinstance(old_classifier.parent, ComparisonPredicate) or isinstance
            if old_classifier.parent.classifier_true == old_classifier:
                old_classifier.parent.classifier_true = new_classifier
            else:
                assert old_classifier.parent.classifier_false == old_classifier, "%s parent=%s" % (old_classifier, old_classifier.parent)
                old_classifier.parent.classifier_false = new_classifier

    def multi_label_entropy(self, prefix, tests):
        """ Given prefix and set of tests, calculates multi-label entropy of tests. """
        cnt = {}
        num_tests = len(tests)
        ret = 0

        for test in tests:
            good_strats = self.gt_classifier.get_good_strats(test, prefix)
            if good_strats is None:
                good_strats = ['None']
            for t in good_strats:
                cnt[t] = cnt[t] + 1 if t in cnt else 1

        for x in cnt.values():
            if x == 0 or x == num_tests:
                continue
            prob = float(x) / num_tests
            assert 0 <= x and x <= num_tests
            ret += prob * np.log(prob) + (1 - prob) * np.log(1 - prob)
        return -ret

    def split_entropy(self, prefix, dataset, classifier):
        """ Given dataset of formulas and classifier, calculates entropy after splitting on predicate. """

        true_tests, false_tests = [], []
        if isinstance(classifier, ComparisonPredicate):
            for _, test in dataset:
                if classifier.eval_predicate(test):
                    true_tests.append(test)
                else:
                    false_tests.append(test)
        else:
            for _, test in dataset:
                true_tests.append(test)

        if len(true_tests) == 0 or len(false_tests) == 0:
            return None

        h_true = self.multi_label_entropy(prefix, true_tests)
        h_false = self.multi_label_entropy(prefix, false_tests)

        return float(len(true_tests)) / len(dataset) * h_true + float(len(false_tests)) / len(dataset) * h_false

    def get_split_opt_score(self, prefix, classifier, candidate_tactics, dataset, validation):
        classifier.reset_stats()

        correct = 0
        score = 0

        for _, test in dataset:
            predicted_tactic = classifier.classify(test, candidate_tactics)
            assert predicted_tactic is not None

            if validation:
                test_score = self.prefix_scorer_valid.get_best_for_prefix(test, prefix, predicted_tactic)
            else:
                test_score = self.prefix_scorer_train.get_best_for_prefix(test, prefix, predicted_tactic)

            score += test_score
            if test_score < TIMEOUT:
                correct += 1

        return score

    def compute_classifier_score_opt(self, classifier, candidate_tactics, dataset, validation=False):
        """ Given a classifier and dataset of formulas, computes the score for that classifier on the given
        set of formulas.

        :param classifier: classifier to compute the score of
        :param candidate_tactics: candidate tactics among which classifier will choose from
        :param dataset: dataset of formulas to classify
        :param validation: whether dataset corresponds to validation or training set
        :return: score of the classifier
        """
        prefix = MultiProgramSynthesizer.get_prefix(classifier)
        score = self.get_split_opt_score(prefix, classifier, candidate_tactics, dataset, validation)

        if not validation:
            classifier.score = score
        return score

    def compute_classifier_score(self, classifier, candidate_tactics, dataset, validation=False):
        if isinstance(classifier, DefaultClassifierModel):
            return self.compute_classifier_score_opt(classifier, candidate_tactics, dataset, validation)
        if isinstance(classifier, ComparisonPredicate) or isinstance(classifier, BoolPredicate):
            return self.compute_classifier_score_entropy(classifier, candidate_tactics, dataset, validation)
        assert False, 'Classifier type unknown!'

    def find_improving_classifiers(self, old_classifier, probe_type, candidate_tactics):
        """ Returns an ordered list of classifiers that have better score than the 'old_classifier' """
        prefix = MultiProgramSynthesizer.get_prefix(old_classifier)

        old_classifier.collect_dataset = True
        self.eval_train.evaluate(self.strategy_tree)
        # Dataset contains all the test evaluated at this branch as well as the optimal tactic to choose
        training_dataset = old_classifier.dataset
        print('\t\t\tTraining Dataset size: ', len(training_dataset))

        self.eval_valid.evaluate(self.strategy_tree)
        validation_dataset = old_classifier.dataset
        print('\t\t\tValidation Dataset size: ', len(validation_dataset))
        old_classifier.collect_dataset = False

        probes = list(self.probe_gen.gen_probes(candidate_tactics, old_classifier, probe_type, prefix, validation_dataset))
        print('\t\t\tFinding best classifier of type %s with %d candidate tactics' % (probe_type, len(candidate_tactics)))
        print('\t\t\tNumber of Candidate Predicates:', len(probes))

        old_validation_score = self.compute_classifier_score(old_classifier, candidate_tactics, validation_dataset, validation=True)

        scored_classifiers = []
        for classifier in probes:
            new_validation_score = self.compute_classifier_score(classifier, candidate_tactics, validation_dataset,
                                                                 validation=True)
            if new_validation_score is None:
                continue
            if ((probe_type == ProbeType.Predicate and isinstance(old_classifier, DefaultClassifierModel)) or
                        new_validation_score < old_validation_score):
                scored_classifiers.append((new_validation_score, len(scored_classifiers), classifier))

        scored_classifiers.sort()
        print('\t\t\tFound %d classifiers that improve the score' % (len(scored_classifiers)))
        for score, _, classifier in scored_classifiers[:5]:
            print('\t\t\t\t', score, classifier)
            assert not math.isnan(float(score))

        return scored_classifiers

    def compute_classifier_score_entropy(self, classifier, candidate_tactics, dataset, validation=False):
        """ Given a classifier and dataset of formulas, computes the score for that classifier on the given
        set of formulas.

        :param classifier: classifier to compute the score of
        :param candidate_tactics: candidate tactics among which classifier will choose from
        :param dataset: dataset of formulas to classify
        :param validation: whether dataset corresponds to validation or training set
        :return: score of the classifier
        """
        prefix = MultiProgramSynthesizer.get_prefix(classifier)
        score = self.split_entropy(prefix, dataset, classifier)

        if not validation:
            classifier.score = score
        return score

    def evaluate_classifier(self, classifier):
        print('\t\tOverall Training Score:', self.eval_train.evaluate(self.strategy_tree))
        print('\t\tTraining Branch Stats:', classifier.stats())
        print('\t\tOverall Validation Score:', self.eval_valid.evaluate(self.strategy_tree))
        print('\t\tValidation Branch Stats:', classifier.stats())

    def synthesize_branch(self, classifier, candidate_tactics, probe_type):
        assert isinstance(classifier, DefaultClassifierModel)
        print('\tSynthesizing Predicate for Leaf Node: ', classifier)
        self.evaluate_classifier(classifier)

        if classifier.total == 0:
            print('\t\t\tPruning, Branch too small with %d samples' % (classifier.total))
            return

        candidate_classifiers = self.find_improving_classifiers(classifier, probe_type, candidate_tactics)
        tree_pos = self.get_parent(classifier)

        for score, _, candidate_classifier in candidate_classifiers:
            print('\t\t\tApply Classifier', candidate_classifier)
            MultiProgramSynthesizer.replace_classifier(classifier, candidate_classifier)
            print('Eval of candidate:')
            self.evaluate_classifier(candidate_classifier)

            if isinstance(candidate_classifier, ComparisonPredicate):
                print(candidate_classifier.true, candidate_classifier.false)
            
            if (isinstance(candidate_classifier, ComparisonPredicate) or
                  isinstance(candidate_classifier, BoolPredicate)):
                if candidate_classifier.true == 0 or candidate_classifier.false == 0:
                    print('Continue ',candidate_classifier.true,candidate_classifier.false)
                    continue
                
                self.synthesize_branch(candidate_classifier.classifier_true, candidate_tactics, ProbeType.Default)
                self.synthesize_branch(candidate_classifier.classifier_false, candidate_tactics, ProbeType.Default)

                if candidate_classifier.true > self.leaf_size:
                    self.synthesize_branch(candidate_classifier.classifier_true, candidate_tactics, ProbeType.Predicate)
                if candidate_classifier.false > self.leaf_size:
                    self.synthesize_branch(candidate_classifier.classifier_false, candidate_tactics, ProbeType.Predicate)
            break

    def synthesize_baseline(self):
        """ Synthesizes baseline which, at each decision point, selects fixed tactic with best score. """
        print('')
        for i, tree_pos in enumerate(self.strategy_tree.get_classifier_positions()):
            print('Optimizing Baseline Branch:', i)
            candidate_tactics = list(tree_pos.tree.keys())
            self.synthesize_branch(tree_pos.classifier, candidate_tactics, ProbeType.Default)

        print('Computing Baseline Score')
        print('\tBaseline Score:', self.eval_train.evaluate(self.strategy_tree))
        print('\tPredicates Score Valid:', self.eval_valid.evaluate(self.strategy_tree))
        print('\tPredicates Score Valid Blind:', self.eval_valid_blind.evaluate(self.strategy_tree))
        self.strategy_tree.print()

    def synthesize_predicates(self):
        """Synthesizes predicates by choosing predicate with largest score at each node."""
        for i, tree_pos in enumerate(self.strategy_tree.get_classifier_positions()):
            print('')
            print('Optimizing Branch:', i)
            candidate_tactics = list(tree_pos.tree.keys())
            self.synthesize_branch(tree_pos.classifier, candidate_tactics, ProbeType.Predicate)

        print('\tPredicates Score Train:', self.eval_train.evaluate(self.strategy_tree))
        print('\tPredicates Score Valid:', self.eval_valid.evaluate(self.strategy_tree))
        print('\tPredicates Score Valid Blind:', self.eval_valid_blind.evaluate(self.strategy_tree))

    def synthesize_predicates_with_smt(self):
        """Synthesizes predicates by choosing predicate with largest score at each node."""
        for i, tree_pos in enumerate(self.strategy_tree.get_classifier_positions()):
            print('')
            print('Optimizing Branch:', i)
            candidate_tactics = list(tree_pos.tree.keys())
            self.synthesize_branch(tree_pos.classifier, candidate_tactics, ProbeType.SMT)

        print('\tPredicates Score Train:', self.eval_train.evaluate(self.strategy_tree))
        print('\tPredicates Score Valid:', self.eval_valid.evaluate(self.strategy_tree))
        print('\tPredicates Score Valid Blind:', self.eval_valid_blind.evaluate(self.strategy_tree))

