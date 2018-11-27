import argparse
import glob
import logging
import numpy as np
import os
import random
import torch
from fastsmt.synthesis.multi.synthesizer import *
from fastsmt.utils.tester import BenchmarkGoalTester

TIMEOUT = 100000000
LOG = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Process Android Constraint Layout Sources.')
    parser.add_argument('--benchmark_dir', type=str, help='Directory with benchmarks')
    parser.add_argument('--cache_file', type=str, default=None, help='File with cached results')
    parser.add_argument('--max_timeout', type=float, default=10, help='Maximum runtime allowed to solving each formular')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--log', type=str, default='INFO', help='Level of logging that should be used')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use during the synthesis')
    parser.add_argument('--leaf_size', type=int, default=10, help='Leaf size')
    parser.add_argument('--num_valid', type=int, default=None, help='Number of validation strategies')
    parser.add_argument('--num_strategies', type=int, default=10, help='Number of strategies to use')
    parser.add_argument('--input_file', type=str, default=None, help='File in which every line represents one candidate strategy')
    parser.add_argument('--strategy_file', type=str, default=None, help='File where synthesized strategy should be stored')
    parser.add_argument('--f_lambda', type=float, default=0.5, help='Weight for solving previously unsolved formula')
    args = parser.parse_args()
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=args.log,
                        format='%(name)s %(levelname)s:    %(message)s')

    # ensure deterministic re-runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    smt_instances_train = glob.glob(os.path.join(os.path.abspath(args.benchmark_dir), 'train', '*.smt2'))
    smt_instances_valid = glob.glob(os.path.join(os.path.abspath(args.benchmark_dir), 'valid', '*.smt2'))

    if args.num_valid is not None:
        random.shuffle(smt_instances_valid)
        smt_instances_valid = smt_instances_valid[:args.num_valid]

    LOG.info('Number of train instances: ' + str(len(smt_instances_train)))
    LOG.info('Number of valid instances: ' + str(len(smt_instances_valid)))

    assert len(smt_instances_train) > 0
    assert len(smt_instances_valid) > 0

    smt_tester = BenchmarkGoalTester(num_threads=args.num_threads)
    if args.cache_file:
        smt_tester.load_cache(args.cache_file)
        smt_tester.out_file = args.cache_file

    LOG.info('Saving cache to ' + smt_tester.out_file)

    strategies = []
    with open(args.input_file, 'r') as f:
        for line in f:
            strategies.append(line[:-1])
    print('Number of strategies: ',len(strategies))
    strategies = list(set(strategies))
    print('Number of unique strategies: ',len(strategies))

    syn = MultiProgramSynthesizer(strategies, smt_tester, smt_instances_train, smt_instances_valid, args.max_timeout, args.leaf_size, args.num_strategies, args.f_lambda)
    #smt_tester.save()

    syn.synthesize_baseline()
    syn.synthesize_predicates()
    syn.strategy_tree.print()
    
    final_strategy = syn.strategy_tree.get_smt2()
    with open(args.strategy_file, 'w') as f:
        f.write(final_strategy)
    
    syn.close()

if __name__ == '__main__':
    main()
