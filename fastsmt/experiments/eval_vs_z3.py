import argparse
import glob
import logging
import os
import numpy as np
import threading
import time
import subprocess
import shlex
from fastsmt.utils.test import BenchmarkTest
from fastsmt.utils.tester import BenchmarkGoalTester
from fastsmt.synthesis.search.models import TrainedModel
from fastsmt.language.objects import from_string

PER = [0.1, 0.5, 0.9]
RND = np.random.randint(10*9)
K = 4

class Z3Runner(threading.Thread):
    """ Runner which executes a single tactic on a single goal. """

    def __init__(self, smt_file, timeout, strategy=None, id=1):
        threading.Thread.__init__(self)
        self.smt_file = smt_file
        self.timeout = timeout
        self.strategy = strategy

        if self.strategy is not None:
            self.tmp_file = open('tmp/tmp_run_{}_{}.smt2'.format(RND,id), 'w')
            with open(self.smt_file, 'r') as f:
                for line in f:
                    new_line = line
                    if 'check-sat' in line:
                        new_line = '(check-sat-using %s)\n' % strategy
                    self.tmp_file.write(new_line)
            self.tmp_file.close()
            self.new_file_name = 'tmp/tmp_run_{}_{}.smt2'.format(RND,id)
        else:
            self.new_file_name = self.smt_file

    def run(self):
        self.time_before = time.time()
        z3_cmd = 'z3 -smt2 %s -st' % self.new_file_name
        self.p = subprocess.Popen(shlex.split(z3_cmd), stdout=subprocess.PIPE)
        self.p.wait()
        self.time_after = time.time()

    def collect(self):
        if self.is_alive():
            try:
                self.p.terminate()
                self.join()
            except OSError:
                pass
            return None, None, None

        out, err = self.p.communicate()

        lines = out[:-1].decode("utf-8").split('\n')
        res = lines[0]

        rlimit = None
        for line in lines:
            if 'rlimit' in line:
                tokens = line.split(' ')
                for token in tokens:
                    if token.isdigit():
                        rlimit = int(token)

        if res != 'unsat' and res != 'sat':
            res = None

        return res, rlimit, self.time_after - self.time_before


def main():
    parser = argparse.ArgumentParser(description='Compare list of strategies against Z3 default strategy.')
    parser.add_argument('--strategies_file', type=str, help='File with all learned strategies')
    parser.add_argument('--benchmark_dir', type=str, help='Directory with benchmarks')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to run with')
    parser.add_argument('--max_timeout', type=int, help='Timeout for each run')
    parser.add_argument('--cache_file', type=str, default=None, help='Cache file')
    parser.add_argument('--log', type=str, default='INFO', help='Level of logging that should be used')
    parser.add_argument('--formulas_batch', type=int, default=1, help='Number of benchmarks to evaluate in parallel')
    parser.add_argument('--strategies_batch', type=int, default=1, help='Number of strategies to evaluate in parallel')
    parser.add_argument('--max_formulas', type=int, default=None, help='Maximum number of formulas to evaluate.')
    parser.add_argument('--load_state', type=str, default=None, help='File from which current state should be loaded')
    args = parser.parse_args()

    logging.basicConfig(level=args.log,
                        format='%(name)s %(levelname)s:    %(message)s')

    smt_instances = glob.glob(os.path.join(os.path.abspath(args.benchmark_dir), '*.smt2'))
    tester = BenchmarkGoalTester(num_threads=args.num_threads, tmp_dir='tmp/')

    if args.cache_file is not None:
        tester.load_cache(args.cache_file)
        tester.out_file = args.cache_file

    strategies = []
    with open(args.strategies_file, 'r') as f:
        for line in f:
            strategies.append(line[:-1])
    print('Number of strategies: ', len(strategies))
    strategies = list(set(strategies))
    smt2_strats = [from_string(x).to_smt2() for x in strategies]
    print('Number of unique strategies: ', len(smt2_strats))

    print(args.max_formulas)
    if args.max_formulas is not None:
        smt_instances = smt_instances[:args.max_formulas]

    print('Number of smt formulas: ',len(smt_instances))
    ok_z3 = {}
    ok_model = {}
    if args.load_state is not None:
        with open(args.load_state, 'r') as f:
            for i, line in enumerate(f):
                tokens = line[:-1].split()
                z3_res, z3_rlimit, z3_time = tokens[1], tokens[2], tokens[3]
                model_res, model_rlimit, model_time = tokens[5], tokens[6], tokens[7]
                if z3_res != 'None':
                    ok_z3[smt_instances[i]] = (model_res, int(z3_rlimit), float(z3_time))
                if model_res != 'None':
                    ok_model[smt_instances[i]] = (model_res, int(model_rlimit), float(model_time))
    
    for it in range(0,1000):
        curr_timeout = min(args.max_timeout, K**it)
        print('------------> Running with timeout ',curr_timeout)
        unsolved_z3 = [instance for instance in smt_instances if instance not in ok_z3]
        unsolved_model = [instance for instance in smt_instances if instance not in ok_model]
        print('Solving total SMT instances: Z3: ',len(unsolved_z3),', Model: ',len(unsolved_model))

        print('Evaluating Z3')
        z3_batch_size = args.formulas_batch * args.strategies_batch
        for i in range(0, len(unsolved_z3), z3_batch_size):
            t_start = time.time()
            batch_instances = unsolved_z3[i:min(i+z3_batch_size, len(unsolved_z3))]
            threads = []
            for instance in batch_instances:
                threads.append(Z3Runner(instance, curr_timeout, None))
            for thread in threads:
                thread.start()
            t1 = time.time()
            for thread in threads:
                t2 = time.time()
                thread.join(max(0.0001,curr_timeout-(t2-t1)))
            for thread in threads:
                res, rlimit, runtime = thread.collect()
                if res is not None and (res == 'sat' or res == 'unsat'):
                    ok_z3[thread.smt_file] = (res, rlimit, runtime)
            t_end = time.time()
            print('Evaluated batch of %d instances [time = %.2f seconds]' % (len(batch_instances),t_end-t_start))
        
        print('Evaluating model strategies')
        for i in range(0, len(unsolved_model), args.formulas_batch):
            batch_instances = unsolved_model[i:min(i+args.formulas_batch, len(unsolved_model))]

            t_start = time.time()
            best_res, best_rlimits, best_runtime = {}, {}, {}
            for j in range(0, len(smt2_strats), args.strategies_batch):
                ta = time.time()
                threads = []
                st_batch = smt2_strats[j:min(len(smt2_strats), j+args.strategies_batch)]
                itr = 0
                for strat in st_batch:
                    for instance in batch_instances:
                        itr += 1
                        threads.append(Z3Runner(instance, curr_timeout, strat, id=itr))
                for thread in threads:
                    thread.start()

                t1 = time.time()
                for thread in threads:
                    t2 = time.time()
                    thread.join(max(0.0001, curr_timeout-(t2-t1)))

                tb = time.time()

                itr = 0
                for strat in st_batch:
                    for instance in batch_instances:
                        res, rlimit, runtime = threads[itr].collect()
                        itr += 1

                        if res is None or (res != 'sat' and res != 'unsat'):
                            continue
                        if instance not in best_rlimits or best_rlimits[instance] > rlimit:
                            best_rlimits[instance] = rlimit
                            best_res[instance] = res
                            best_runtime[instance] = runtime

            for instance in batch_instances:
                if instance in best_res:
                    ok_model[instance] = (best_res[instance], best_rlimits[instance], best_runtime[instance])
            t_end = time.time()
            print('Evaluated batch of %d instances [time = %.2f seconds]' % (len(batch_instances),t_end-t_start))


        ok_both, ok_none = 0, 0
        only_z3, only_model = 0, 0
        speedups = []
        speedups_runtime = []
        for instance in smt_instances:
            if instance in ok_model:
                model_res, model_rlimit, model_runtime = ok_model[instance]
            else:
                model_res, model_rlimit, model_runtime = None, None, None

            if instance in ok_z3:
                z3_res, z3_rlimit, z3_runtime = ok_z3[instance]
            else:
                z3_res, z3_rlimit, z3_runtime = None, None, None

            print(instance, ' Z3: ',z3_res, z3_rlimit, z3_runtime, '\tLearned:', model_res, model_rlimit, model_runtime)

            z3_solved = (z3_res == 'sat' or z3_res == 'unsat') and z3_rlimit > 0
            model_solved = (model_res == 'sat' or model_res == 'unsat') and model_rlimit > 0

            if z3_solved and model_solved:
                assert z3_res == model_res
                ok_both += 1
                speedup = z3_rlimit / float(model_rlimit)
                speedup_runtime = z3_runtime / float(model_runtime)
                speedups.append(speedup)
                speedups_runtime.append(speedup_runtime)
            elif z3_solved:
                only_z3 += 1
            elif model_solved:
                only_model += 1
            else:
                ok_none += 1

        print('==========================================')
        print('Both solved:',ok_both)
        print('Only learned solved: ',only_model)
        print('Only Z3 solved: ',only_z3)
        print('None solved: ',ok_none)
        if len(speedups) > 0:
            speedups.sort()
            print('Average speedup: ',np.mean(speedups))
            for p in PER:
                print('Percentile ',p,': ',speedups[int(p*len(speedups))])
        if len(speedups_runtime) > 0:
            speedups_runtime.sort()
            print('[T] Average speedup: ',np.mean(speedups_runtime))
            for p in PER:
                print('[T] Percentile ',p,': ',speedups_runtime[int(p*len(speedups_runtime))])
        if curr_timeout >= args.max_timeout:
            break


if __name__ == '__main__':
    main()
