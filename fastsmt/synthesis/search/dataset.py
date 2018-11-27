import logging
import numpy as np
import pickle
import random


class Sample:

    def __init__(self, tactics, target, target_params, target_probs):
        self.tactics = tactics
        self.target = target
        self.target_params = target_params
        self.target_probs = target_probs


class FeaturizedSample(Sample):

    def __init__(self, tactics, target, target_params, target_probs, features):
        super(FeaturizedSample, self).__init__(tactics, target, target_params, target_probs)
        self.features = features


class Dataset:

    def __init__(self, samples=[]):
        self.samples = samples
        self.n_samples = len(self.samples)

    def add_sample(self, sample):
        self.samples.append(sample)
        self.n_samples += 1

    def get_tactics(self):
        return np.array([sample.tactics for sample in self.samples])

    def get_target_params(self):
        return [sample.target_params for sample in self.samples]

    def get_train_targets(self):
        return np.array([sample.target for sample in self.samples])

    def get_asts(self):
        """ Returns array of ASTs associated with samples, samples need to be ASTSample. """
        assert all([isinstance(sample, ASTSample) for sample in self.samples]), 'Some samples are not ASTSample!'
        return [sample.get_ast() for sample in self.samples]

    def get_features(self):
        """ Returns numpy array of features associated with samples, samples need to be FeaturizedSample. """
        assert all([isinstance(sample, FeaturizedSample) for sample in self.samples]), 'Some samples are not FeaturizedSample!'
        return np.array([sample.features for sample in self.samples])

    def get_target_probs(self, num_tactics):
        """ Returns n_samples x num_tactics matrix with ones at positions of target tactics. """
        train_target_probs = np.zeros((self.n_samples, num_tactics))
        for i, sample in enumerate(self.samples):
            train_target_probs[i] = sample.target_probs
        return train_target_probs

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Dataset([self.samples[i] for i in range(item.start, item.stop)])
        elif isinstance(item, list):
            return Dataset([self.samples[i] for i in item])
        else:
            raise TypeError('Unknown type for splitting samples collection: ' + str(type(item)))

    def split_train_valid(self, train_percent):
        indices = [i for i in range(self.n_samples)]
        random.shuffle(indices)

        first_valid_idx = int(self.n_samples * train_percent)
        train_indices = indices[:first_valid_idx]
        valid_indices = indices[first_valid_idx:]
        return self[train_indices], self[valid_indices]


