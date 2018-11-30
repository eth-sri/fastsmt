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

import random
import json
import logging
import os
import tempfile
import time

try:
    import fastText
except ImportError:
    logging.warn('fastText could not be imported (only needed to run fastText model).')

from abc import ABC, abstractmethod
from sklearn.externals import joblib

from fastsmt.synthesis.search.dataset import *
from fastsmt.utils.strategy import StrategyEnumerator
from fastsmt.language.objects import *
from fastsmt.synthesis.search_strategies import ScoredCandidateStatus
from fastsmt.synthesis.search.neural_nets import *
from fastsmt.utils.utilities import evaluate_candidate_strategy, evaluate_candidate_strategies

MAX_LEN = 20
INF = 10**10

class Model(ABC):

    def __init__(self, config):
        self.config = config
        if "tactics_config" in config:
            tactics_config = config["tactics_config"]
        else:
            tactics_config = {}

        self.strategy_enum = StrategyEnumerator(**tactics_config)

    def retrain(self):
        pass

    def reset(self):
        pass

    def add_scored_strategy(self, scored_candidate, status):
        pass

    def load_model(self, file):
        pass

    def save_model(self, file):
        pass

    def can_predict(self):
        return True

    def predict_arguments(self, tactic, parent=None):
        if tactic not in self.strategy_enum.param_max:
            return {}
        return {param: np.random.random() for param in self.strategy_enum.param_max[tactic]}

    @abstractmethod
    def score_strategy(self, strategy, parent=None):
        pass


class RandomModel(Model):
    """ Model which assigns random score to strategy. """

    def score_strategy(self, strategy, parent=None):
        return np.log(random.random())

class BFSModel(Model):
    """ Model which assigns score to the strategy based on its length. """

    def score_strategy(self, strategy, parent=None):
        return -len(strategy)

    def predict_arguments(self, tactic, parent=None):
        if tactic not in self.strategy_enum.param_max:
            return {}
        return {param: np.random.random() for param in self.strategy_enum.param_max[tactic]}


class FastTextModel(Model):
    """ Bilinear model which assigns score to strategy."""

    def __init__(self, config):
        super(FastTextModel, self).__init__(config)
        self.log = logging.getLogger('FastTextModel')
        self.data = []
        self.bilinear_model = None
        self.explore_rate = 1.0

        self.discretized_tactics = {}
        for tactic in self.strategy_enum.all_tactics:
            if tactic not in self.strategy_enum.allowed_params:
                self.discretized_tactics[tactic] = [{}]
                continue
            
            self.discretized_tactics[tactic] = []
            for _ in range(self.config['models']['fast_text']['discretize']):
                disc_tac = {param: np.random.randint(2) for param in self.strategy_enum.param_max[tactic]}
                self.discretized_tactics[tactic].append(disc_tac)

    def map_to_discretized(self, tactic, params):
        for i, disc_params in enumerate(self.discretized_tactics[tactic]):
            match = True
            for param in params:
                if int(disc_params[param]) != int(params[param]):
                    match = False
            if match:
                return i
        assert False, 'Found no match!'

    def encode(self, tactics):
        if len(tactics) == 0:
            return 'NULL'
        return ' '.join([str(t.s) for t in tactics])

    def can_predict(self):
        return True

    def load_model(self, file):
        self.bilinear_model = fastText.load_model(file + '.bin')
        print('Loaded bilinear model from ',file)

    def save_model(self, file):
        self.bilinear_model.save_model(file + '.bin')

    def retrain(self):
        if len(self.data) < self.config['models']['fast_text']['min_train_data']:
            return

        self.log.info('Retraining bilinear model, explore_rate = %.2f' % self.explore_rate)

        train_file = tempfile.NamedTemporaryFile(mode='w')

        best_tactic = {}
        for scored_candidate, status in self.data:
            smt_file = scored_candidate.benchmarks[0].file
            tactics = get_tactics(scored_candidate.t)

            if status != ScoredCandidateStatus.SOLVED:
                continue

            for i, tactic in enumerate(tactics):
                fast_text_line = self.encode(tactics[:i])
                key = (smt_file, fast_text_line)
                if key not in best_tactic or best_tactic[key][1] > scored_candidate.rlimit:
                    best_tactic[key] = (tactic.s, scored_candidate.rlimit)
                    entry = '__label__%s %s' % (tactic.s, fast_text_line)
                    train_file.write(entry + '\n')

        for scored_candidate, status in self.data:
            smt_file = scored_candidate.benchmarks[0].file
            tactics = get_tactics(scored_candidate.t)

            if status != ScoredCandidateStatus.SOLVED:
                continue

            for i, tactic in enumerate(tactics):
                params = self.strategy_enum.extract_params([tactic])[0]
                disc_idx = self.map_to_discretized(tactic.s, params)
                fast_text_line = self.encode(tactics[:i])
                key = (smt_file, fast_text_line)
                
                if best_tactic[key][1] == scored_candidate.rlimit:
                    entry = '__label__%s_%d %s' % (tactic.s, disc_idx, fast_text_line)
                    train_file.write(entry + '\n')
        train_file.flush()
        os.fsync(train_file.fileno())

        self.log.info('Created dataset of %d entries' % len(best_tactic))

        self.bilinear_model = fastText.train_supervised(
            input=train_file.name,
            epoch=self.config['models']['fast_text']['epoch'],
            lr=self.config['models']['fast_text']['lr'],
            wordNgrams=self.config['models']['fast_text']['ngrams'],
            verbose=self.config['models']['fast_text']['verbose'],
            minCount=self.config['models']['fast_text']['min_count'],
            dim=self.config['models']['fast_text']['dim'],
        )

    def score_strategy(self, strategy, parent=None):
        if self.bilinear_model is None or np.random.random() < self.explore_rate:
            return np.log(random.random())

        parent_tactics = self.encode(get_tactics(parent.t))
        target_label = '__label__' + strategy[-1].s

        labels, probs = self.bilinear_model.predict(parent_tactics, 1000)
        ret_score = 0
        found = False
        for label, prob in zip(labels, probs):
            if label.startswith(target_label):
                score = np.log(prob) + parent.score
                ret_score += score
                found = True

        if not found:
            return -INF
        return ret_score

    def predict_arguments(self, tactic, parent=None):
        if tactic not in self.strategy_enum.param_max:
            return {}
        if tactic not in self.strategy_enum.allowed_params:
            return {}
        if self.bilinear_model is None or np.random.random() < self.explore_rate:
            idx = np.random.randint(len(self.discretized_tactics[tactic]))
            return self.discretized_tactics[tactic][idx]

        parent_tactics = self.encode(get_tactics(parent.t))

        #print('Predicting arg for ',tactic)

        labels, probs = self.bilinear_model.predict(parent_tactics, 1000)
        best = None
        for i in range(len(self.discretized_tactics[tactic])):
            target_label = '__label__' + tactic + '_' + str(i)
            for label, prob in zip(labels, probs):
                if label == target_label:
                    score = np.log(prob) + parent.score
                    #print(label, ' -> best: ', best)
                    if best is None or (score, i) > best:
                        best = (score, i)

        if best is None:
            idx = np.random.randint(len(self.discretized_tactics[tactic]))
        else:
            idx = best[1]
        return self.discretized_tactics[tactic][idx]

    def add_scored_strategy(self, scored_candidate, status):
        self.data.append((scored_candidate, status))
        self.explore_rate *= self.config['exploration']['explore_decay']
        self.explore_rate = np.minimum(self.explore_rate, self.config['exploration']['min_explore_rate'])
        

class ApprenticeModel(Model):
    """ Model which uses neural network to guide the search.
    It is trained using Dagger. """

    def __init__(self, config):
        super(ApprenticeModel, self).__init__(config)
        self.config = config
        if "tactics_config" in self.config:
            tactics_config = self.config["tactics_config"]
        else:
            tactics_config = {}

        self.type = self.config['models']['apprentice']['type']
        self.num_tactics = len(self.strategy_enum.base_tactics)
        self.tac_idx = {str(tac): i
                        for i, tac in enumerate(self.strategy_enum.base_tactics)}

        self.log = logging.getLogger('ApprenticeModel')
        self.data = []
        self.trained = False
        self.num_train = 0

        self.nn = None
        if self.type == 'ast':
            self.init_network()

        self.mem_predict = None
        self.param_sigma = 0.3
        self.valid = False
        self.trained = False
        self.model_data = None

    def can_predict(self):
        return self.trained

    def load_now(self):
        assert self.nn is not None
        self.nn.load_state_dict(torch.load(self.model_data['net']))
        self.nn.scaler = joblib.load(self.model_data['net'] + '_scaler.joblib')
        self.log.info('Loaded model and scaler!')

    def load_model(self, path):
        self.model_data = {
            'net': path,
        }
        if self.nn is not None:
            self.load_now()
    
    def save_model(self, path):
        if self.trained:
            torch.save(self.nn.state_dict(), path)
            joblib.dump(self.nn.scaler, path + '_scaler.joblib')

    def reset(self):
        pass

    def init_network(self, num_features=None):
        """ Initializes neural network."""
        params_per_tactic = self.strategy_enum.get_params_per_tactic()
        if self.type == 'bow':
            self.nn = PolicyNN(self.num_tactics, num_features, params_per_tactic, 30, MAX_LEN)
            if self.model_data is not None:
                self.load_now()
        elif self.type == 'ast':
            # TODO: Check for number of tokens here
            self.nn = TreeNN(self.num_tactics, 300, params_per_tactic, MAX_LEN, 10, 10)
        else:
            assert False, 'Unknown feature type: ' + self.type

    def featurize_tactics(self, tactics):
        """ Given list of tactics, returns featurized representation. """
        if len(tactics) > MAX_LEN:
            tactics = tactics[-MAX_LEN:]
        ret = [self.num_tactics for _ in range(MAX_LEN)]
        if tactics[0] == 'Tactic(skip)':
            tactics = tactics[1:]

        for i, tactic in enumerate(tactics):
            ret[i] = self.tac_idx[tactic]
        return ret

    def featurize_params(self, tactics):
        """ Given list of tactics, extracts parameters from each of them. """
        return self.strategy_enum.extract_params(tactics)

    def featurize_candidate(self, strategy, to_numpy=True):
        """ Featurizes the candidate, considering both applied tactics an features of the formula. """
        tactics = get_tactics(strategy.t)
        params = self.featurize_params(tactics)
        tactics = self.featurize_tactics(['Tactic(%s)' % tac.s for tac in tactics])
        if to_numpy:
            tactics = np.array(tactics)

        if self.type == 'bow':
            bow = strategy.get_bow()[0]
            probes = strategy.get_probes()[0]
            features = np.array(bow)

            if to_numpy:
                probes = np.array(probes)
                features = np.array(features)

            features = np.concatenate((probes, features))

            return tactics, features, params
        else:
            assert False, 'Unknown feature type: ' + self.type

    def get_target_probs(self, rlimit_per_action, best_rlimit):
        probs = np.zeros(len(self.strategy_enum.base_tactics))
        for action in rlimit_per_action:
            probs[action] = best_rlimit / rlimit_per_action[action][2]
        probs /= np.sum(probs)
        return probs

    def retrain(self):
        """ Retrain the network using all collected data. """
        self.log.debug('Datapoints collected: ' + str(len(self.data)))

        best_per_file = {}
        for scored_candidate, status in self.data:
            if status != ScoredCandidateStatus.SOLVED:
                continue
            smt_file = scored_candidate.benchmarks[0].file
            if smt_file not in best_per_file or best_per_file[smt_file] < scored_candidate.rlimit:
                best_per_file[smt_file] = scored_candidate.rlimit

        rlimit_per_action = {}
        best_tactic = {}
        for scored_candidate, status in self.data:
            if status != ScoredCandidateStatus.SOLVED:
                continue
            smt_file = scored_candidate.benchmarks[0].file
            tactics, features, all_params = self.featurize_candidate(
                scored_candidate, to_numpy=False)

            if scored_candidate.rlimit > 2 * best_per_file[smt_file]:
                continue

            for i, (tactic, params) in enumerate(zip(tactics, all_params)):
                if tactic == self.num_tactics:
                    break
                key = (smt_file, ' '.join(map(str,tactics[:i])))

                if key not in best_tactic or best_tactic[key][3] > scored_candidate.rlimit:
                    best_tactic[key] = (tactic, self.strategy_enum.base_tactics[tactic].s, params, scored_candidate.rlimit)
                if key not in rlimit_per_action:
                    rlimit_per_action[key] = {}
                rlimit_per_action[key][tactic] = (self.strategy_enum.base_tactics[tactic].s, params, scored_candidate.rlimit)

        self.log.debug('Calculated %d best tactics',len(best_tactic))

        dataset = Dataset()

        for scored_candidate, status in self.data:
            if status != ScoredCandidateStatus.SUCCESS:
                continue
            ast = None
            smt_file = scored_candidate.benchmarks[0].file

            if self.type == 'bow':
                tactics, features, params = self.featurize_candidate(scored_candidate, to_numpy=False)
            else:
                tactics, ast, params = self.featurize_candidate(scored_candidate, to_numpy=False)
                # It is possible that AST is none because computation was terminated
                if ast is None:
                    continue

            end_pos = -1
            for i, tactic in enumerate(tactics):
                if tactic == self.num_tactics:
                    end_pos = i
                    break
            if end_pos == -1:
                continue

            key = (smt_file, ' '.join(map(str, tactics[:end_pos])))
            if key not in best_tactic:
                continue

            target_probs = self.get_target_probs(rlimit_per_action[key], best_tactic[key][3])
            target_idx = best_tactic[key][0]
            target_params = (best_tactic[key][1], best_tactic[key][2])

            if self.type == 'bow':
                dataset.add_sample(FeaturizedSample(tactics, target_idx, target_params, target_probs, features))
            else:
                dataset.add_sample(ASTSample(tactics, target_idx, target_params, ast))

        if dataset.n_samples < self.config['models']['apprentice']['min_train_data']:
            print('Data size = %d is too low, not training' % dataset.n_samples)
            return

        self.nn.retrain(self.config, dataset)
        self.trained = True
        if len(self.data) > 10000:
            self.data = self.data[-10000:]

    def add_scored_strategy(self, scored_candidate, status):
        self.log.debug('num_train: %s %d' % (str(self.num_train), self.trained))
        self.log.debug('Scored strategy: ' + str(scored_candidate) + ' ' + str(status))

        if status == ScoredCandidateStatus.SUCCESS or status == ScoredCandidateStatus.SOLVED:
            self.data.append((scored_candidate, status))

        tactics, features, params = self.featurize_candidate(scored_candidate)
        if self.nn is None and self.type == 'bow':
            self.init_network(features.shape[0])

    # TODO: Maybe cache here, will be called 20x for same parent.
    def score_strategy(self, strategy, parent=None):
        key = (parent.benchmarks[0].file, parent.get_goal_hashes()[0][0])
        idx = self.tac_idx['Tactic(%s)' % str(strategy[-1].s)]

        if len(strategy) >= MAX_LEN:
            return -1000000000000

        if self.type == 'ast':
            if self.mem_predict is None or self.mem_predict[0] != key:
                tactics, ast, params = self.featurize_candidate(parent, to_numpy=False)

                # change this to be prob distribution
                if ast is None:
                    return np.random.randn()

                sample = ASTSample(tactics, None, None, ast)
                log_probs = self.nn.predict(sample)
                self.mem_predict = (key, log_probs[0][0])

            score = self.mem_predict[1][idx].item()
            return score + parent.score

        tactics, features, params = self.featurize_candidate(parent)
        if self.nn is None and self.type == 'bow':
            self.init_network(features.shape[0])

        features = self.nn.scaler.transform(features.reshape(1, -1)).reshape(-1)        
        probs, log_probs = self.nn.predict(tactics, features)
        ret = log_probs[idx]
        self.log.debug('nn score: ' + str(ret+parent.score))
        return ret + parent.score

    def predict_arguments(self, tactic, parent=None):
        if tactic not in self.strategy_enum.param_max or len(self.strategy_enum.param_max[tactic]) == 0:
            return {}

        tactics, features, params = self.featurize_candidate(parent)

        if self.nn is None and self.type == 'bow':
            self.init_network(features.shape[0])

        params_pred = self.nn.predict_params(tactic, tactics, features)
        ret = {param: np.clip(np.random.normal(params_pred[0][i].item(), self.param_sigma), 0, 1)
               for i, param in enumerate(self.strategy_enum.param_max[tactic])}
        return ret


class TrainedModel:
    """ Loads model with already synthesized strategies. """

    def __init__(self, name, model, strategies_file, config_file):
        if name == 'default':
            self.name = name
            self.model = None
            self.strategies = [None]
            self.config = None
        else:
            self.name = name
            self.model = model
            self.strategies = TrainedModel.load_synthesized_strategies(strategies_file)
            self.config = TrainedModel.load_config(config_file)
        self.log = logging.getLogger('TrainedModel')

    def get_model(self):
        return self.model

    @staticmethod
    def load_config(file):
        if file is None:
            return None
        return json.load(open(file, 'r'))

    def get_config(self):
        return self.config

    @staticmethod
    def load_synthesized_strategies(file):
        synthesized_strategies = []
        if not file or not os.path.exists(file):
            return synthesized_strategies
        with open(file, 'r') as f:
            for line in f:
                synthesized_strategies.append(from_string(line.strip()))
        return synthesized_strategies

    def get_synthesized_strategies(self):
        return self.strategies

    def solve_instance(self, tester, smt_instance, max_timeout):
        tasks = evaluate_candidate_strategies(tester, self.strategies, smt_instance, max_timeout)
        best_task = None
        for task in tasks:
            if task.is_solved() and (best_task is None or best_task.rlimit > task.rlimit):
                best_task = task
        return best_task

    def solve_instances_batched(self, tester, smt_instances, max_timeout, batch_size):
        strategies = self.get_synthesized_strategies()
        self.log.info('Evaluating strategy ' + str(strategies[0]))
        best_tasks = [None for _ in smt_instances]

        for i in range(0, len(strategies), batch_size):
            batch_strategies = strategies[i:min(len(strategies), i+batch_size)]
            evaled_tasks = evaluate_candidate_strategies(tester, batch_strategies, smt_instances, max_timeout, best_tasks)
            for j, task in evaled_tasks:
                if best_tasks[j] is None:
                    best_tasks[j] = task
                    continue
                if task.is_solved() and (not best_tasks[j].is_solved() or best_tasks[j].rlimit > task.rlimit):
                    best_tasks[j] = task
        tester.save_cache()
        return best_tasks


    def solve_instances(self, tester, smt_instances, max_timeout, all=False):
        strategies = self.get_synthesized_strategies()
        self.log.info('Evaluating strategy ' + str(strategies[0]))
        best_tasks = evaluate_candidate_strategy(tester, strategies[0], smt_instances, max_timeout)
        all_tasks = []
        if all:
            for i, task in enumerate(best_tasks):
                all_tasks.append(task)

        for strategy in strategies[1:]:
            self.log.info('Evaluating strategy ' + str(strategy))
            tasks = evaluate_candidate_strategy(tester, strategy, smt_instances, max_timeout, best_tasks)
            for i, task in enumerate(tasks):
                all_tasks.append(task)
                if task.is_solved() and (not best_tasks[i].is_solved() or best_tasks[i].rlimit > task.rlimit):
                    best_tasks[i] = task
        tester.save_cache()

        if all:
            return best_tasks, all_tasks

        return best_tasks
