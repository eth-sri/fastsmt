import logging
import numpy as np
import time
import torch
import torchfold
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import sys
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam
from fastsmt.synthesis.search.dataset import *

sys.setrecursionlimit(1500)

EMBED_SIZE = 30
MAX_DEPTH = 20

#TODO: Add eps noise to softmax to prevent NaN
#TODO: Maybe move batch-norm after activation
class PolicyNN(nn.Module):
    """ Neural network model which is given features of the formula and tactics
    applied to it so far and returns probability distribution over next
    tactics to apply. """

    def __init__(self, num_tactics, num_features, params_per_tactic, embed_size, max_len):
        """ Initializes object of type PolicyNN.

        :param num_tactics: number of tactics available
        :param num_features: number of features of the formula
        :param params_per_tactic: number of parameters that need to be supplied for each tactic
        :param embed_size: size of the embedding for each tactic
        :param max_len: maximum length of the strategy
        """
        super(PolicyNN, self).__init__()
        self.num_tactics = num_tactics
        self.num_features = num_features
        self.embed_size = embed_size
        self.max_len = max_len

        self.embedding = nn.Embedding(num_tactics + 1, self.embed_size)

        self.fc0 = nn.Linear(self.max_len * self.embed_size, 100)
        self.bn0 = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100 + 100, 50)
        self.bn1 = nn.BatchNorm1d(50)

        self.fc2 = nn.Linear(50, num_tactics)
        self.bn2 = nn.BatchNorm1d(num_tactics)
        self.param_layer = {}
        for tactic, num_params in params_per_tactic.items():
            if num_params == 0:
                continue
            self.param_layer[tactic] = nn.Linear(50, num_params)

        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.fc3 = nn.Linear(self.num_features, 100)
        self.bn3 = nn.BatchNorm1d(100)

        self.fc_value = nn.Linear(50, 1)
        self.log = logging.getLogger('PolicyNN')

    def encoder(self, tactics, features):
        """ Given tactics applied so far and features describing the formula this function encodes them in a vector.
        :param tactics: tactics applied so far
        :param features: features describing the formula
        :return: K-dimensional vector which is result of encoding
        """
        tactics = self.embedding(tactics)
        tactics = tactics.view(-1, self.max_len * self.embed_size)
        tactics = F.relu(self.bn0(self.fc0(tactics)))

        y = F.relu(self.bn3(self.fc3(features)))
        z = torch.cat([tactics, y], dim=1)
        z = F.relu(self.bn1(self.fc1(z)))

        return z

    def forward_params(self, tactic, encoder_out):
        """ Given tactic and output of encoder, predicts parameters of the tactic. """
        return torch.sigmoid(self.param_layer[tactic](encoder_out))

    def forward(self, tactics, features):
        """ Calculates output of feed-forward neural network.

        :param tactics: sequence of tactics applied with padding so that its length is max_len
        :param features: features extracted from the formula (such as bag-of-words)
        :return: probabilities and log-probabilities for each tactic
        """
        z = self.encoder(tactics, features)
        logits = self.bn2(self.fc2(z))

        probs = self.softmax(logits)
        log_probs = self.log_softmax(logits)

        return z, probs, log_probs

    def predict(self, tactics, features, batch_size=1):
        """ Given samples, runs inference. """
        self.eval()
        _, probs, log_probs = self.forward(
            Variable(torch.from_numpy(tactics.reshape(batch_size, -1)).long()),
            Variable(torch.from_numpy(features.reshape(batch_size, -1)).float()),
        )
        return probs.view(-1).detach().numpy(), log_probs.view(-1).detach().numpy()

    def predict_params(self, new_tactic, tactics, features):
        self.eval()
        enc_out = self.encoder(Variable(torch.from_numpy(tactics.reshape(1, -1)).long()),
                         Variable(torch.from_numpy(features.reshape(1, -1)).float()))
        params = self.forward_params(new_tactic, enc_out)
        return params

    def validate(self, n_valid, tactics, features, target_probs, target_params):
        """ Receives set of samples and runs inference on them.

        :param n_valid: number of samples to validate
        :param tactics: tactics associated with samples
        :param features: features associated with samples
        :param target_probs: target probabilities associated with samples
        :param target_params: target parameters for each tactic in the sample
        :return:
        """
        self.eval()

        encoder_out, _, valid_log_probs = self.forward(
            Variable(torch.from_numpy(tactics).long()),
            Variable(torch.from_numpy(features).float()),
        )
        valid_kl_loss = -torch.dot(
            valid_log_probs.view(-1),
            Variable(torch.from_numpy(target_probs).float()).view(-1)) / n_valid
    
        #for each tactic, predict arguments
        valid_mse_loss = torch.FloatTensor([0])
        for j in range(n_valid):
            tactic_name, true_params = target_params[j]
            if tactic_name not in self.param_layer:
                assert len(true_params) == 0
                continue
            pred_params = self.forward_params(tactic_name, encoder_out[j])
            true_params = torch.FloatTensor([value for _, value in true_params.items()])
            valid_mse_loss += torch.dot(pred_params - true_params, pred_params - true_params)
        valid_mse_loss /= n_valid

        return valid_kl_loss.item(), valid_mse_loss.item()

    def preprocess(self, train, valid):
        self.scaler = StandardScaler()
        feats = train.get_features()
        self.scaler.fit(feats)

        for sample in train.samples:
            sample.features = self.scaler.transform(sample.features.reshape(1,-1)).reshape(-1)
        for sample in valid.samples:
            sample.features = self.scaler.transform(sample.features.reshape(1,-1)).reshape(-1)

        return train, valid

    def retrain(self, config, dataset):
        """ Retrains the network. """
        self.share_memory()
        self.config = config
        adam_lr = config['models']['apprentice']['adam_lr']
        num_epochs = config['models']['apprentice']['epochs']
        mini_batch_size = config['models']['apprentice']['mini_batch_size']
        min_valid_samples = config['models']['apprentice']['min_valid_samples']
        valid_split = config['models']['apprentice']['valid_split']

        if dataset.n_samples < mini_batch_size or mini_batch_size > dataset.n_samples:
            self.log.warning('Number of samples = %d too low to train the policy' % dataset.n_samples)
            return

        if dataset.n_samples >= min_valid_samples:
            train, valid = dataset.split_train_valid(valid_split)
        else:
            train, valid = dataset, Dataset([])

        self.log.info('Training dataset size: %d' % train.n_samples)
        self.log.info('Validation dataset size: %d' % valid.n_samples)

        self.writer = SummaryWriter()

        train, valid = self.preprocess(train, valid)
        train_tactics, valid_tactics = train.get_tactics(), valid.get_tactics()
        train_features, valid_features = train.get_features(), valid.get_features()
        train_target_params, valid_target_params = train.get_target_params(), valid.get_target_params()
        train_target_probs, valid_target_probs = train.get_target_probs(self.num_tactics), valid.get_target_probs(self.num_tactics)

        optimizer = Adam(self.parameters(), lr=adam_lr)

        self.log.info('Training the apprentice on %d samples' % dataset.n_samples)

        prev_valid_kl_loss = -1
        valid_inc = 0

        best_valid_loss = None
        best_state = None

        epoch = 0
        while True:
            epoch += 1

            if valid.n_samples == 0 and epoch > num_epochs:
                break

            self.train()
            idx = torch.randperm(train.n_samples)

            all_mse_loss = []
            all_kl_loss = []

            for i in range(0, train.n_samples - mini_batch_size + 1, mini_batch_size):
                optimizer.zero_grad()

                mini_batch = Variable(idx[i:i + mini_batch_size])
                batch_tactics = train_tactics[mini_batch, :]
                batch_features = train_features[mini_batch, :]
                batch_target_params = [train_target_params[idx[i + j]] for j in range(mini_batch_size)]

                encoder_out, probs, log_probs = self.forward(
                    Variable(torch.from_numpy(batch_tactics).long()),
                    Variable(torch.from_numpy(batch_features).float()),
                )

                # predict probability distribution over tactics
                probs, log_probs = probs.view(-1), log_probs.view(-1)

                batch_target_probs = train_target_probs[mini_batch, :]
                batch_target_probs = Variable(
                    torch.from_numpy(batch_target_probs).float()).view(-1)

                kl_loss = -torch.dot(log_probs, batch_target_probs) / mini_batch_size

                # for each tactic, predict arguments
                mse_loss = torch.FloatTensor([0])
                for j in range(mini_batch_size):
                    tactic_name, true_params = batch_target_params[j]
                    if tactic_name not in self.param_layer:
                        assert len(true_params) == 0
                        continue
                    pred_params = self.forward_params(tactic_name, encoder_out[j])
                    true_params = torch.FloatTensor([value for _, value in true_params.items()])
                    mse_loss += torch.dot(pred_params - true_params, pred_params - true_params) / mini_batch_size

                total_loss = 0.1 * kl_loss + 0.9 * mse_loss
                total_loss.backward()

                all_kl_loss.append(kl_loss.item())
                all_mse_loss.append(mse_loss.item())
                self.log.debug('Train KL loss: ' + str(kl_loss.item()))
                self.log.debug('Train MSE loss: ' + str(mse_loss.item()))
                self.log.debug('Total train loss: ' + str(total_loss.item()))

                optimizer.step()

            self.writer.add_scalar('train_parameters_mse_loss', np.mean(all_mse_loss), epoch)
            self.writer.add_scalar('train_tactics_kl_loss', np.mean(all_kl_loss), epoch)

            if epoch % 10 == 0:
                self.log.info('Epoch: %d, Train CE loss = %.6f, Train MSE loss = %.6f' % (
                    epoch, kl_loss.item(), mse_loss.item()))

            if valid.n_samples == 0 or (epoch + 1) % 10 > 0:
                continue

            valid_kl_loss, valid_mse_loss = self.validate(
                valid.n_samples, valid_tactics, valid_features, valid_target_probs, valid_target_params)
            total_valid_loss = valid_kl_loss + valid_mse_loss

            self.writer.add_scalar('valid_parameters_mse_loss', valid_mse_loss, epoch)
            self.writer.add_scalar('valid_tactics_kl_loss', valid_kl_loss, epoch)
            self.writer.add_scalar('valid_tactics_total_loss', total_valid_loss, epoch)

            if (best_valid_loss is None) or best_valid_loss > total_valid_loss:
                best_valid_loss = total_valid_loss
                best_state = self.state_dict()
                self.log.info('Saving state, new best valid score: ' + str(best_valid_loss))

            if valid_kl_loss > prev_valid_kl_loss:
                valid_inc += 1
            else:
                valid_inc = 0
            prev_valid_kl_loss = valid_kl_loss

            if epoch % 10 == 0:
                self.log.info('Valid CE loss: %s [inc = %d]' % (str(valid_kl_loss), valid_inc))

            if valid_inc == config['models']['apprentice']['early_stopping_inc']:
                break

        if best_state is not None:
            self.log.info('Training done, restoring best found model...')
            self.load_state_dict(best_state)
