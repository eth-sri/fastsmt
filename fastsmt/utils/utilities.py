import logging
import z3
from fastsmt.utils.test import *
from fastsmt.utils.tester import BenchmarkGoalTester

LOG = logging.getLogger(__name__)

def evaluate_candidate_strategy(tester, strategy, smt_instances, max_timeout, best_tasks=None):
    """
    Given strategy and list of smt instances, evaluates the strategy on instances and
    reports back evaluated tasks.

    :param tester: BenchmarkGoalTester object to be used for evaluation
    :param strategy: strategy to test
    :param smt_instances: smt instances on which to test the strategy
    :param max_timeout: maximum time limit allowed for evaluation
    :param best_tasks: current best tasks per formula
    :return: evaluated tasks
    """
    LOG.debug('Evaluating strategy %s on %d instances!' % (str(strategy), len(smt_instances)))
    if strategy is None:
        tasks = [BenchmarkTest(smt_instance, strat=None, timeout=max_timeout)
                 for smt_instance in smt_instances]
    else:
        tasks = []
        for i, smt_instance in enumerate(smt_instances):
            timeout = max_timeout
            if best_tasks is not None and best_tasks[i].is_solved():
                timeout = best_tasks[i].runtime
            if timeout < 1.0:
                timeout = 1.0
            tasks.append(IncrementalBenchmarkTest(smt_instance, strat=strategy, timeout=timeout))
    tester.evaluate_parallel(tasks)
    return tasks

def evaluate_candidate_strategies(tester, strategies, smt_instances, max_timeout, best_tasks=None):
    """
    Given strategy and list of smt instances, evaluates the strategy on instances and
    reports back evaluated tasks.

    :param tester: BenchmarkGoalTester object to be used for evaluation
    :param strategies: strategies to test
    :param smt_instances: smt instances on which to test the strategy
    :param max_timeout: maximum time limit allowed for evaluation
    :param best_tasks: current best tasks per formula
    :return: evaluated tasks
    """
    LOG.info('Evaluating %d strategies on %d instances!' % (len(strategies), len(smt_instances)))
    if strategies is None:
        tasks = [BenchmarkTest(smt_instance, strat=None, timeout=max_timeout)
                 for smt_instance in smt_instances]
        ids = list(range(len(tasks)))
    else:
        tasks = []
        ids = []
        for strategy in strategies:
            for i, smt_instance in enumerate(smt_instances):
                timeout = max_timeout
                if best_tasks is not None and best_tasks[i] is not None and best_tasks[i].is_solved():
                    timeout = best_tasks[i].runtime + 0.5
                if timeout < 1.0:
                    timeout = 1.0
                tasks.append(IncrementalBenchmarkTest(smt_instance, strat=strategy, timeout=timeout))
                ids.append(i)
                
    tester.evaluate_parallel(tasks)
    return zip(ids, tasks)


"""
Converts formula f to SMT2 format string.
"""
def toSMT2Benchmark(f, status="unknown", name="benchmark", logic=""):
    v = (z3.Ast * 0)()
    return z3.Z3_benchmark_to_smtlib_string(f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast())

"""
Converts string s to goal.
"""
def goal_from_string(s):
    f = z3.parse_smt2_string(s)
    ret_goal = z3.Goal()
    ret_goal.add(f)
    return ret_goal

"""
Parses SMT2 file and returns resulting goal.
"""
def goal_from_file(filename):
    f = z3.parse_smt2_file(filename)
    ret_goal = z3.Goal()
    ret_goal.add(f)
    return ret_goal
