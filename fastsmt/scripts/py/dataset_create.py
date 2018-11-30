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
import os
import random
import shutil

def create_dir(all_dir, target_dir, files):
    for file in files:
        target_filepath = os.path.join(target_dir, file)
        source_filepath = os.path.join(all_dir, file)
        os.symlink(source_filepath, target_filepath)

def main():
    parser = argparse.ArgumentParser(description='Create train/valid/test dataset from files in the folder.')
    parser.add_argument('--split', type=str, required=True, help='Percentage of formulas that should go into train/valid/test')
    parser.add_argument('--benchmark_dir', type=str, required=True, help='Benchmark folder')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    train, valid, test = list(map(int, args.split.split(' ')))
    assert train + valid + test == 100, 'Split percentages should add up to 100!'

    train_dir = os.path.abspath(os.path.join(args.benchmark_dir, 'train'))
    valid_dir = os.path.abspath(os.path.join(args.benchmark_dir, 'valid'))
    test_dir = os.path.abspath(os.path.join(args.benchmark_dir, 'test'))

    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
    if os.path.isdir(valid_dir):
        shutil.rmtree(valid_dir)
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)

    os.mkdir(train_dir)
    os.mkdir(valid_dir)
    os.mkdir(test_dir)

    all_dir = os.path.abspath(os.path.join(args.benchmark_dir, 'all'))
    assert os.path.exists(all_dir), 'Files should be present in all/ subdirectory!'

    for root, directories, filenames in os.walk(all_dir):
        all_files = filenames.copy()
        random.shuffle(all_files)

        n_samples = len(all_files)
        print('Total files: ',n_samples)

        train_size = int(float(train) / 100 * n_samples)
        valid_size = int(float(valid) / 100 * n_samples)
        test_size = int(float(test) / 100 * n_samples)

        create_dir(all_dir, train_dir, all_files[:train_size])
        create_dir(all_dir, valid_dir, all_files[train_size:train_size+valid_size])
        create_dir(all_dir, test_dir, all_files[train_size+valid_size:])

        break


if __name__ == '__main__':
    main()
