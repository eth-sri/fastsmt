import logging

import numpy as np
from abc import ABCMeta, abstractmethod
from enum import Enum
from fastsmt.language.objects import *

class PredicateOp(Enum):
    GT = 1
    LT = 2
    EQ = 3

class ArithmeticOp(Enum):
    DIV = 1
    MUL = 2
    ADD = 3
    SUB = 4

OPS_DICT = {
    PredicateOp.EQ: '=',
    PredicateOp.LT: '<',
    PredicateOp.GT: '>',
}

class ClassifierModelBase:
    """ Base class for classifier which determines which branch to take for each formula. """
    __metaclass__ = ABCMeta

    def __init__(self, parent):
        """ Initializes the classifier.

        :param parent: parent of the classifier
        """
        self.parent = parent
        self.correct = 0
        self.total = 0
        self.collect_dataset = False
        self.dataset = []

    def reset_stats(self):
        """ Resets all statistics measured inside of classifier. """
        self.correct = 0
        self.total = 0
        self.dataset = []

    def entropy(self):
        """ Calculates the entropy of classifier predictions. """
        if self.correct == 0 or self.correct == self.total:
            return 0
        true_ratio = self.correct / self.total
        false_ratio = (self.total - self.correct) / self.total
        return - true_ratio * np.log2(true_ratio) - false_ratio * np.log2(false_ratio)

    def stats(self):
        """ Returns stats measured inside of the classifier. """
        return "correct %d, total %d" % (self.correct, self.total)

    def get_leaf_nodes(self):
        return [self]

    def get_tree(self):
        assert False
        # if isinstance(self.parent, StrategyTree):
        #     return self.parent
        # return self.parent.GetTree()

    def classify(self, test, values, synthesized_tactics=None, best_tactics=None):
        """ Classifies the given test case.

        :param test: test that should be classified
        :param values: branches among which classification should be done
        :param synthesized_tactics: prefix consisting of tactics synthesized so far
        :param best_tactics: branches which are considered correct if predicted
        :return: branch which shuold be taken
        """
        if self.collect_dataset:
            self.dataset.append((best_tactics, test))
        predicted_tactic = self.classify_inner(test, values, synthesized_tactics, best_tactics)
        # predicted_tactic is correct if it belongs to best_tactics
        if best_tactics and predicted_tactic in best_tactics:
            self.correct += 1
        self.total += 1
        return predicted_tactic

    @abstractmethod
    def Score(self):
        pass

    @abstractmethod
    def classify_inner(self, test, values, tactics, best_tactics):
        pass


class DefaultClassifierModel(ClassifierModelBase):
    """ Default classifier which always predicts same fixed branch. """

    def __init__(self, tactic_name, parent = None):
        """ Initializes the default classifier.

        :param tactic_name: name of the tactic that should be predicted
        :param parent: parent of the classifier
        """
        super(DefaultClassifierModel, self).__init__(parent)
        self.tactic_name = tactic_name

    def classify_inner(self, test, values, synthesized_tactics, best_tactic):
        assert self.tactic_name in values
        return self.tactic_name

    def Score(self):
        return self.correct

    def to_smt2(self, tree):
        s = from_string(self.tactic_name)
        assert self.tactic_name in tree.keys()

        child_smt2 = tree[self.tactic_name].get_smt2()

        if len(child_smt2) > 0:
            return '(then %s %s)' % (s.to_smt2(), child_smt2)
        else:
            return s.to_smt2()

    def __str__(self):
        return "DefaultClassifierModel[%s]" % (self.tactic_name)

    def __repr__(self):
        return str(self)


class ComparisonPredicate(ClassifierModelBase):

    def __init__(self, parent, classifier_true, classifier_false, predicate_op, probe_name, probe_value):
        super(ComparisonPredicate, self).__init__(parent)

        self.classifier_true = classifier_true
        self.classifier_false = classifier_false

        if self.classifier_true is not None:
            self.classifier_true.parent = self
        if self.classifier_false is not None:
            self.classifier_false.parent = self

        self.op = predicate_op
        assert probe_name is not None
        self.probe_name = probe_name
        self.probe_value = probe_value
        self.reset_stats()

    def reset_stats(self):
        super(ComparisonPredicate, self).reset_stats()
        self.classifier_true.reset_stats()
        self.classifier_false.reset_stats()
        self.true = 0
        self.false = 0

    def eval_predicate(self, test):
        if self.op == PredicateOp.EQ and test.get_probe_value(self.probe_name) == self.probe_value:
            return True
        if self.op == PredicateOp.LT and test.get_probe_value(self.probe_name) < self.probe_value:
            return True
        if self.op == PredicateOp.GT and test.get_probe_value(self.probe_name) > self.probe_value:
            return True
        return False

    def classify_inner(self, test, values, synthesized_tactics, best_tactics):
        if self.eval_predicate(test):
            self.true += 1
            return self.classifier_true.classify(test, values, synthesized_tactics, best_tactics)
        else:
            self.false += 1
            return self.classifier_false.classify(test, values, synthesized_tactics, best_tactics)

    def to_smt2(self, tree):
        if '+' in self.probe_name:
            e1, e2 = self.probe_name.split('+')
            alpha1, probe1 = e1.split('*')
            alpha2, probe2 = e2.split('*')
            alpha1 = int(alpha1)
            alpha2 = int(alpha2)
            name = '(+ (* %d %s) (* %d %s))' % (alpha1, probe1, alpha2, probe2)
        else:
            name = self.probe_name
        print(name)
        cond = '%s %s %d' % (OPS_DICT[self.op], name, self.probe_value)
        true_smt2 = self.classifier_true.to_smt2(tree)
        false_smt2 = self.classifier_false.to_smt2(tree)
        return '(if (%s) %s %s)' % (cond, true_smt2, false_smt2)

    def get_leaf_nodes(self):
        return self.classifier_true.get_leaf_nodes() + self.classifier_false.get_leaf_nodes()

    def __str__(self):
        return "Predicate[%s %s %s] true[%s] false[%s]" % (self.probe_name, self.op, str(self.probe_value), self.classifier_true, self.classifier_false)

    def __repr__(self):
        return str(self)

class ArithmeticComparisonPredicate(ClassifierModelBase):

    def __init__(self, parent, classifier_true, classifier_false, predicate_op, probe_name1, probe_name2, probe_value):
        super(ArithmeticComparisonPredicate, self).__init__(parent)

        self.classifier_true = classifier_true
        self.classifier_false = classifier_false

        if self.classifier_true is not None:
            self.classifier_true.parent = self
        if self.classifier_false is not None:
            self.classifier_false.parent = self

        self.op = predicate_op
        assert probe_name1 is not None
        assert probe_name2 is not None
        self.probe_name1 = probe_name1
        self.probe_name2 = probe_name2
        self.probe_value = probe_value
        self.reset_stats()

    def reset_stats(self):
        super(ComparisonPredicate, self).reset_stats()
        self.classifier_true.reset_stats()
        self.classifier_false.reset_stats()
        self.true = 0
        self.false = 0

    def eval_predicate(self, test):
        if self.op == PredicateOp.EQ and test.get_probe_value(self.probe_name) == self.probe_value:
            return True
        if self.op == PredicateOp.LT and test.get_probe_value(self.probe_name) < self.probe_value:
            return True
        if self.op == PredicateOp.GT and test.get_probe_value(self.probe_name) > self.probe_value:
            return True
        return False

    def classify_inner(self, test, values, synthesized_tactics, best_tactics):
        if self.eval_predicate(test):
            self.true += 1
            return self.classifier_true.classify(test, values, synthesized_tactics, best_tactics)
        else:
            self.false += 1
            return self.classifier_false.classify(test, values, synthesized_tactics, best_tactics)

    def to_smt2(self):
        cond = '%s %s %d' % (self.probe_name, OPS_DICT[self.op], self.probe_value)
        true_smt2 = self.classifier_true.to_smt2()
        false_smt2 = self.classifier_false.to_smt2()
        return '(if (%s) %s %s)' % (cond, true_smt2, false_smt2)

    def get_leaf_nodes(self):
        return self.classifier_true.get_leaf_nodes() + self.classifier_false.get_leaf_nodes()

    def __str__(self):
        return "Predicate[%s %s %s] true[%s] false[%s]" % (self.probe_name, self.op, str(self.probe_value), self.classifier_true, self.classifier_false)

    def __repr__(self):
        return str(self)


class BoolPredicate(ClassifierModelBase):

    def __init__(self, parent, classifier_true, classifier_false, probe_name):
        super(BoolPredicate, self).__init__(parent)

        self.classifier_true = classifier_true
        self.classifier_false = classifier_false

        if self.classifier_true is not None:
            self.classifier_true.parent = self
        if self.classifier_false is not None:
            self.classifier_false.parent = self

        self.probe_name = probe_name
        self.reset_stats()

    def reset_stats(self):
        super(BoolPredicate, self).reset_stats()
        self.classifier_true.reset_stats()
        self.classifier_false.reset_stats()
        self.true = 0
        self.false = 0

    def eval_predicate(self, test):
        return bool(test.get_probe_value(self.probe_name))

    def classify_inner(self, test, values, synthesized_tactics, best_tactics):
        if self.eval_predicate(test):
            self.true += 1
            return self.classifier_true.classify(test, values, synthesized_tactics, best_tactics)
        else:
            self.false += 1
            return self.classifier_false.classify(test, values, synthesized_tactics, best_tactics)

    def to_smt2(self):
        cond = self.probe_name
        true_smt2 = self.classifier_true.to_smt2()
        false_smt2 = self.classifier_false.to_smt2()
        return '(if %s %s %s)' % (cond, true_smt2, false_smt2)

    def get_leaf_nodes(self):
        return self.classifier_true.get_leaf_nodes() + self.classifier_false.get_leaf_nodes()

    def __str__(self):
        return "Predicate[%s] true[%s] false[%s]" % (self.probe_name, self.classifier_true, self.classifier_false)

    def __repr__(self):
        return str(self)


class ClassifierSMT(ClassifierModelBase):
    """ Classifier which determines to which tactic does the SMT formula map based on
    logical formulas constructed from probes. """

    def __init__(self, parent, probe_names, probe_bounds, mapping):
        """
        Initializes object of type ClassifierSMT.

        :param parent: parent node in the tree
        :param probe_names: names of the probes used by classifier
        :param probe_bounds: bounds for probes used by classifier
        :param mapping: mapping from probes names to evaluations for each tactic
        """
        super(ClassifierSMT, self).__init__(parent)
        self.parent = parent
        self.probe_names = probe_names
        self.probe_bounds = probe_bounds
        self.mapping = mapping
        self.log = logging.getLogger('ClassifierSMT')

    def eval(self, test):
        """ Evaluates test and returns binary mask of evaluations on probes. """
        ret = {}
        for pname, pbound in zip(self.probe_names, self.probe_bounds):
            ret[pname] = (test.get_probe_value(pname) < pbound)
        return ret

    def classify_inner(self, test, values, tactics, best_tactics):
        for value in values:
            assert value in self.mapping, 'Value not in mapping!'

        evals = self.eval(test)

        cnt_ok = {}
        for tactic in values:
            cnt_ok[tactic] = 0
            for key, val in self.mapping[tactic].items():
                if evals[key] == val:
                    cnt_ok[tactic] += 1
            cnt_ok[tactic] /= float(len(self.mapping[tactic].items()))

        sorted_tactics = sorted(cnt_ok.items(), key=lambda x: -x[1])

        return sorted_tactics[0][0]


class TryForPredicate(ClassifierModelBase):

    def __init__(self, parent, classifier_false, tactic_name, timeout):
        super(TryForPredicate, self).__init__(parent)
        self.classifier_false = classifier_false
        self.classifier_false.parent = self
        self.tactic_name = tactic_name
        self.timeout = timeout

    def reset_stats(self):
        super(TryForPredicate, self).reset_stats()
        self.classifier_false.ResetStats()

    def GetLeafNodes(self):
        return self.classifier_false.GetLeafNodes()

    def classify_inner(self, test, values, synthesized_tactics, best_tactics):
        tree = self.get_tree()
        try_test = tree.GetTest(test.file, get_tactics(test.strat) + [from_string(self.tactic_name)])
        if try_test.res != 'fail' and try_test.runtime - test.runtime < self.timeout:
            return self.tactic_name
        else:
            return self.classifier_false.Classify(test, values, synthesized_tactics, best_tactics)

    def __str__(self):
        return "TryForPredicate[%s %f]" % (self.tactic_name, self.timeout)

    def __repr__(self):
        return str(self)

class GroundTruthClassifierModel:

    def __init__(self, threshold=1.1):
        self.best_task_per_smt = {}
        self.good_strats = {}
        self.threshold = threshold
        self.cb = None

    def add_tasks(self, all_tasks):
        for task in all_tasks:
            tactics = get_tactics(task.strat)

            if task.res != 'sat' and task.res != 'unsat':
                continue

            for i in range(len(tactics)):
                prefix = ' '.join(list(map(str, tactics[:i])))
                best_tactic = tactics[i]
                if (task.file, prefix) not in self.best_task_per_smt:
                    self.best_task_per_smt[(task.file, prefix)] = []

                # append all tasks that solve the instnace

                self.best_task_per_smt[(task.file, prefix)].append(task)

        for key in self.best_task_per_smt:
            self.best_task_per_smt[key].sort(key=lambda task: task.rlimit)
            best_rlimit = self.best_task_per_smt[key][0].rlimit
            self.good_strats[key] = [
                task.strat
                for task in self.best_task_per_smt[key]
                if task.rlimit <= best_rlimit * self.threshold
            ]
            self.good_strats[key] = list(set(self.good_strats[key]))

    def get_good_strats(self, test, synthesized_tactics=None):
        prefix = ' '.join(map(str, synthesized_tactics[1:]))
        if (test.file, prefix) not in self.good_strats:
            return None
        return self.good_strats[(test.file, prefix)]

    def stats(self):
        return "--"

    def classify(self, test, values, synthesized_tactics=None, best_tactics=None):
        assert synthesized_tactics
        prefix = ' '.join(map(str, synthesized_tactics[1:]))
        if (test.file, prefix) not in self.best_task_per_smt:
            return None

        best_tasks = self.best_task_per_smt[(test.file, prefix)]
        best_tactics = []

        for best_task in best_tasks:
            best_tactic = get_tactics(best_task.strat)[len(synthesized_tactics) - 1]
            assert str(best_tactic) in values
            best_tactics.append(str(best_tactic))

        if self.cb:
            self.cb.Add(test, best_tactics)

        return list(set(best_tactics))
