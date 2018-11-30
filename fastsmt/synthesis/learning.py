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

import argparse
import json
import random
import logging
import shutil
import os
import numpy as np
import time
import torch

from fastsmt.synthesis.search_strategies import ModelSearch
from fastsmt.synthesis.search.models import ApprenticeModel, FastTextModel, RandomModel, BFSModel
from fastsmt.synthesis.evo_strategy import EvoSearch
from fastsmt.utils.tester import BenchmarkGoalTester
from fastsmt.utils.test import IncrementalBenchmarkTest, BenchmarkTest

NUM_PROBES = 12


class Synthesizer(object):
    """ Object which is wrapper over synthesis procedure. It is initialized with a custom config
    which contains information on the benchmark, maximum time limit and models to be used
    for synthesis. """

    def __init__(self, config, pop_size, max_timeout, experiment_name, eval_dir=None):
        """ Initializes object of type Synthesizer.

        :param config: dict with configuration of the synthesis procedure
        :param pop_size: size of the population
        :param max_timeout: maximum allowed runtime for each test
        :param experiment_name: name of the experiment
        :param eval_dir: directory where results of evaluation will be stored
        """
        self.log = logging.getLogger('Synthesizer')
        self.config = config
        self.pop_size = pop_size
        self.max_timeout = max_timeout
        self.experiment_name = experiment_name
        self.eval_dir = eval_dir
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)

        self.main_model = self.get_model(self.config['main_model'])
        self.explore_model = self.get_model(self.config['explore_model'])
        self.smt_instances = {}
        self.num_valids = 0
        self.num_train = 0

        self.run_dir = os.path.join(self.eval_dir, self.experiment_name)
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)

        self.models_dir = os.path.join(self.run_dir, 'models')
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        self.best_candidate = {}

    def save_candidates(self, name, all_candidates, smt_instance, valid):
        """ Saves synthesized strategies to a file. """
        if self.eval_dir is None:
            self.log.warning('Not saving candidates, eval directory not given!')
            return

        if valid:
            candidates_dir = os.path.join(self.eval_dir, name, 'valid', str(self.num_valids))
        else:
            candidates_dir = os.path.join(self.eval_dir, name, 'train', str(self.num_train))
        if not os.path.exists(candidates_dir):
            os.makedirs(candidates_dir)

        smt_file = smt_instance[str.rfind(smt_instance, '/')+1:]
        log_file = os.path.join(candidates_dir, smt_file + '.log')

        with open(log_file, 'w') as f:
            for candidate in all_candidates:
                if candidate is None:
                    print('None', file=f)
                else:
                    print(json.dumps(candidate.benchmarks[0].to_json()), file=f)

        best_file = os.path.join(candidates_dir, 'strategies.txt')
        with open(best_file, 'w') as f:
            for smt_file, best_strategy in self.best_candidate.items():
                if best_strategy is None:
                    continue
                print(best_strategy.t, file=f)

        best_file_compact = os.path.join(candidates_dir, 'strategies_compact.txt')
        with open(best_file_compact, 'w') as f:
            for smt_file, best_strategy in self.best_candidate.items():
                if best_strategy is None:
                    continue
                print(best_strategy, file=f)
                    
    def get_model(self, model_name):
        """ Creates model object from config, used to guide the search. """
        if model_name == "random":
            return RandomModel(self.config)
        elif model_name == "bfs":
            return BFSModel(self.config)
        elif model_name == 'apprentice':
            return ApprenticeModel(self.config)
        elif model_name == 'fast_text':
            return FastTextModel(self.config)

        assert False, 'Model not found {}'.format(model_name)

    def load_dataset(self, benchmark_dir, type):
        """ Reads training or validation dataset from train_seq.txt file in benchmark directory """
        self.smt_instances[type] = []

        for root, directories, filenames in os.walk(benchmark_dir):
            for file in filenames:
                if file.endswith('smt2'):
                    self.smt_instances[type].append(os.path.join(root, file))

        logging.info('Dataset of type [%s] loaded, smt_instances: %d' % (type, len(self.smt_instances[type])))

    def load_tester(self, cache_file, num_threads, tmp_dir):
        """ Creates BenchmarkGoalTester object and loads cache from cache_file. """
        self.benchmark_goal_tester = BenchmarkGoalTester(num_threads=num_threads)

        if cache_file:
            self.benchmark_goal_tester.load_cache(cache_file)
            self.benchmark_goal_tester.out_file = cache_file

    def cleanup_tester(self):
        """ Cleans benchmark goal tester. """
        self.benchmark_goal_tester.cleanup()

    def synthesize_candidate_strategy(self, all_instances, max_timeout, num_iters, valid):
        """ Synthesizes candidate strategy that is best among strategies that can solve all given instances.

        :param all_instances: contains all instances for which strategy should be synthesized
        :param max_timeout: maximum runtime allowed for the strategy
        :param num_iters: number of iterations to run the search for
        :param valid: whether synthesis is training or validation step
        :return: best candidate strategy for each instance
        """
        
        self.search_strategy.restart(all_instances, valid)
        self.log.info('Starting synthesis, number of formulas:' + str(len(all_instances)))

        best_candidate = {smt_instance: None for smt_instance in all_instances}
        solved_candidates = {smt_instance: set() for smt_instance in all_instances}
        all_candidates = {smt_instance: [] for smt_instance in all_instances}

        for it in range(num_iters):
            self.log.info('===================')
            self.log.info('Iteration: %d/%d' % (it+1, num_iters))
            self.log.debug(str(self.search_strategy.get_stats()))
            self.log.info('Strategies evaluated: %d' % self.search_strategy.num_evaluated)
            self.log.info('===================')

            it_start = time.time()

            if it == 0:
                self.search_strategy.init_population(self.pop_size)
            else:
                for smt_instance in all_instances:
                    extend = self.search_strategy.extend_population(self.pop_size, smt_instance)
                    if not extend:
                        continue
            t3 = time.time()
            self.log.debug('Extend population time (s): ' + str(t3 - it_start))
            self.search_strategy.evaluate(max_timeout)

            t1 = time.time()
            for smt_instance in all_instances:
                if smt_instance not in self.best_candidate:
                    self.best_candidate[smt_instance] = None

                candidates = self.search_strategy.get_candidates(smt_instance)
                best_candidate[smt_instance] = self.get_best_candidate(best_candidate[smt_instance], candidates)
                self.best_candidate[smt_instance] = self.get_best_candidate(self.best_candidate[smt_instance], candidates)

                if len(candidates) > 0:
                    all_candidates[smt_instance].append(candidates[0])

                solved_candidates[smt_instance] = solved_candidates[smt_instance].union(
                    self.solved_candidates(self.search_strategy.get_candidates(smt_instance)))

                self.log.debug(smt_instance + " " + str(best_candidate[smt_instance]))
            t2 = time.time()
            self.log.debug('Updating results time (s): ' + str(t2 - t1))

            t1 = time.time()
            if best_candidate is not None:
                tot_solved, tot_unsolved, tot_rlimit = 0, 0, 0
                for smt_file, candidate in best_candidate.items():
                    if candidate is None:
                        tot_unsolved += 1
                    else:
                        tot_solved += 1
                        tot_rlimit += candidate.rlimit
                self.log.info('solved: %d, unsolved: %d, rlimit: %f (m)' % (tot_solved, tot_unsolved, float(tot_rlimit)/10**6))
            t2 = time.time()
            self.log.debug('Collecting best results time (s): ' + str(t2 - t1))
            it_end = time.time()
            self.log.info('Iteration time (s): ' + str(it_end - it_start))

        self.benchmark_goal_tester.save_cache()

        self.log.info('Saving all candidates...')
        for smt_instance in all_instances:
            self.save_candidates(self.experiment_name, all_candidates[smt_instance], smt_instance, valid)
        print('Saved!')

        return best_candidate, solved_candidates

    def evaluate_candidate_strategy(self, candidate, test_smt_instances, max_timeout):
        """ Evaluates candidate strategy on a given set of test SMT instances. """
        if candidate is None:
            tasks = [BenchmarkTest(x[0], strat=None, timeout=max_timeout) for x in test_smt_instances]
        else:
            tasks = [
                IncrementalBenchmarkTest(
                    file=smt_file,
                    strat=candidate.t,
                    timeout=max_timeout)
                for smt_file, _ in test_smt_instances
            ]

        self.benchmark_goal_tester.evaluate_sequential(tasks)

        solved = []
        unsolved = []
        mask = ''

        for i, task in enumerate(tasks):
            assert test_smt_instances[i][0] == task.file

            if task.res == 'sat' or task.res == 'unsat':
                solved.append(test_smt_instances[i])
                mask += '1'
            else:
                unsolved.append(test_smt_instances[i])
                mask += '0'

        print('-------> solved %d, unsolved %d, strategy %s' % (len(solved), len(unsolved), str(candidate.t) if candidate is not None else 'default'))
        return solved, unsolved, mask

    @staticmethod
    def get_best_candidate(best_candidate, candidates):
        best = best_candidate
        for candidate in candidates:
            if candidate.all_solved() and (best is None or best.rlimit > candidate.rlimit):
                best = candidate
        return best

    @staticmethod
    def solved_candidates(candidates):
        res = set()
        for candidate in candidates:
            if candidate.all_solved():
                res.add(str(candidate.t))
        return res

    def synthesize_batch(self, batch, num_iters, valid):
        """ Synthesizes best strategy for each formula in the batch.

        :param batch: list of smt instances for which strategy should be synthesized
        :param num_iters: number of iterations to run the synthesis for
        :param valid: whether batch corresponds to training or validation data
        """
        candidates, _ = self.synthesize_candidate_strategy(
            all_instances=batch,
            max_timeout=self.max_timeout,
            num_iters=num_iters,
            valid=valid)
        if not valid:
            self.main_model.retrain()

    def validation_pass(self, batch_size, num_iters):
        """ Performs one pass over the validation dataset.

        :param batch_size: number of samples in one batch
        :param num_iters: number of iterations to run the synthesis for
        """
        self.num_valids += 1
        for i in range(0, len(self.smt_instances['valid']), batch_size):
            j = min(len(self.smt_instances['valid']), i + batch_size)

            batch = self.smt_instances['valid'][i:j]
            self.synthesize_batch(batch, num_iters, True)
            self.cleanup_tester()

    def training_pass(self, batch_size, num_iters, model_file=None):
        """ Performs one pass over the training dataset.

        :param batch_size: number of samples in one batch
        :param num_iters: number of iterations to run the synthesis for
        :param model_file: file where model should be stored
        """
        logging.info('Starting training pass')
        self.num_train += 1
        for i in range(0, len(self.smt_instances['train']), batch_size):
            j = min(len(self.smt_instances['train']), i + batch_size)

            batch = self.smt_instances['train'][i:j]
            self.synthesize_batch(batch, num_iters, False)
            self.cleanup_tester()

        if model_file is not None:
            self.log.info('Saving model in %s' % model_file)
            self.main_model.save_model(model_file)

    def create_search_strategy(self, evo):
        if evo:
            self.search_strategy = EvoSearch(
                tester=self.benchmark_goal_tester,
                config=self.config,
                model=self.main_model
            )
        else:
            self.search_strategy = ModelSearch(
                tester=self.benchmark_goal_tester,
                config=self.config,
                main_model=self.main_model,
                explore_model=self.explore_model,
            )


def ExistingDir(value):
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError("Directory '" + value + "' does not exist!")
    return value


def main():
    parser = argparse.ArgumentParser(description='Perform synthesis.')
    parser.add_argument('json_config', type=str, help='Json with algorithms configuration')
    parser.add_argument('--benchmark_dir', type=ExistingDir, required=True, help='Benchmark directory')
    parser.add_argument('--cache_file', type=str, default=None, help='File with cached results')
    parser.add_argument('--max_timeout', type=float, default=10, help='Maximum runtime allowed to solving each formular')
    parser.add_argument('--num_iters', type=int, default=10, help='Number of iterations of synthesis')
    parser.add_argument('--pop_size', type=int, default=1, help='Number of strategies in the population')
    parser.add_argument('--iters_inc', type=int, default=10, help='Increase of iterations of synthesis')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log', type=str, default='INFO', help='Level of logging that should be used')
    parser.add_argument('--eval_dir', type=str, required=True, help='Folder where all candidates during synthesis should be stored')
    parser.add_argument('--smt_batch_size', type=int, default=1, help='Number of smt formulas to evaluate in a batch')
    parser.add_argument('--full_pass', type=int, default=1, help='Number of full passes through the entire dataset')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use during the synthesis')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of this experiment run')
    parser.add_argument('--evo', action='store_true', help='Whether to use evolutionary search strategy.')
    parser.add_argument('--tmp_dir', type=str, default='tmp/', help='Temporary directory where SMT formulas should be saved')
    parser.add_argument('--very_small_test', action='store_true', help='Whether to use very small validation set.')
    parser.add_argument('--validate_model', type=str, default=None, help='Whether validation pass should be performed')
    args = parser.parse_args()

    experiment_name = str(int(time.time())) if args.experiment_name is None else args.experiment_name
    valid_name = 'very_small_test' if args.very_small_test else 'valid'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=args.log,
                        format='%(name)s %(levelname)s:    %(message)s')

    config = json.load(open(args.json_config, 'r'))

    if args.validate_model is not None:
        config['exploration']['enabled'] = False

    # ensure deterministic re-runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    synthesizer = Synthesizer(config, args.pop_size, args.max_timeout, experiment_name=experiment_name, eval_dir=args.eval_dir)
    shutil.copy2(args.json_config, synthesizer.run_dir)

    benchmark_dir = os.path.abspath(args.benchmark_dir)
    train_dir = os.path.join(benchmark_dir, 'train')
    valid_dir = os.path.join(benchmark_dir, valid_name)

    train_dir = os.path.abspath(train_dir)
    synthesizer.load_dataset(train_dir, 'train')
    if args.validate_model is not None:
        synthesizer.load_dataset(valid_dir, 'valid')

    synthesizer.load_tester(args.cache_file, args.num_threads, args.tmp_dir)
    synthesizer.create_search_strategy(args.evo)
    num_iters = args.num_iters

    if args.validate_model is not None:
        model_file = os.path.join(synthesizer.models_dir, args.validate_model)
        synthesizer.main_model.load_model(model_file)
        synthesizer.main_model.trained = True
        synthesizer.validation_pass(args.smt_batch_size, args.num_iters)
    else:
        for i in range(args.full_pass):
            model_file = os.path.join(synthesizer.models_dir, 'model_' + str(i+1) + '.pt')
            synthesizer.training_pass(args.smt_batch_size, num_iters, model_file)
            num_iters += args.iters_inc
    synthesizer.benchmark_goal_tester.close()


if __name__ == '__main__':
    main()
