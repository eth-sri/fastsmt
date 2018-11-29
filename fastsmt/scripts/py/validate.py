import argparse
import shlex
import numpy as np
import os
import subprocess
import tempfile
import threading
import time
import z3

PER = [0.1, 0.5, 0.9]
RND = np.random.randint(10**9)

class Z3Runner(threading.Thread):
    """ Runner which executes a single tactic on a single goal. """

    def __init__(self, smt_file, timeout, strategy=None, id=1):
        threading.Thread.__init__(self)
        self.smt_file = smt_file
        self.timeout = timeout
        self.strategy = strategy

        if self.strategy is not None:
            self.tmp_file = open('tmp/tmp_valid_{}_{}.smt2'.format(RND, id), 'w')
            with open(self.smt_file, 'r') as f:
                for line in f:
                    new_line = line
                    if 'check-sat' in line:
                        new_line = '(check-sat-using %s)\n' % strategy
                    self.tmp_file.write(new_line)
            self.tmp_file.close()
            self.new_file_name = 'tmp/tmp_valid_{}_{}.smt2'.format(RND, id)
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

        if res == 'unknown':
            res = None

        return res, rlimit, self.time_after - self.time_before


def main():
    parser = argparse.ArgumentParser(description='Evaluate synthesized strategy')
    parser.add_argument('--strategy_file', type=str, default=None, help='File which contains strategy in SMT2 format')
    parser.add_argument('--benchmark_dir', type=str, help='Directory which contains benchmark files')
    parser.add_argument('--max_timeout', type=int, help='Maximum runtime for solver')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of benchmarks to evaluate in parallel')
    args = parser.parse_args()

    strategy = None
    if args.strategy_file is not None:
        with open(args.strategy_file, 'r') as f:
            strategy = f.readlines()[0]

    ok1 = 0
    okd = 0
    speedups = []
    speedups_real = []

    only_learned = 0
    only_z3 = 0

    for root, directories, filenames in os.walk(args.benchmark_dir):
        for i in range(0, len(filenames), args.batch_size):
            tasks1 = []
            tasks2 = []

            for j in range(i, min(len(filenames), i + args.batch_size)):
                file = filenames[j]
                if not file.endswith('smt2'):
                    continue

                smt_file = os.path.join(args.benchmark_dir, file)

                thread1 = Z3Runner(smt_file, args.max_timeout, strategy, id=j-i+1)
                thread1.start()

                thread2 = Z3Runner(smt_file, args.max_timeout)
                thread2.start()

                tasks1.append(thread1)
                tasks2.append(thread2)

            time_start = time.time()
            for task1, task2 in zip(tasks1, tasks2):
                time_left = max(0, args.max_timeout - (time.time() - time_start))
                task1.join(time_left)
                time_left = max(0, args.max_timeout - (time.time() - time_start))
                task2.join(time_left)

                res1, rlimit1, time1 = task1.collect()
                res2, rlimit2, time2 = task2.collect()
                
                if res1 is not None and res2 is not None and res1 != res2:
                    print(res1, res2)
                    print('Inconsistent result detected, skipping!')
                    print(task1.new_file_name)
                    print(task2.new_file_name)
                    exit(0)
                    continue

                if res1 is not None:
                    if res2 is None:
                        only_learned += 1
                    ok1 += 1
                if res2 is not None:
                    if res1 is None:
                        only_z3 += 1
                    okd += 1

                if res1 is not None and res2 is not None:
                    speedup = rlimit2 / float(rlimit1)
                    speedup_real = time2 / float(time1)
                    speedups.append(speedup)
                    speedups_real.append(speedup_real)

                print('Learned: ',res1, rlimit1, '\tZ3: ', res2, rlimit2)
            print('==========================================')
            print('Both solved:',len(speedups))
            print('Only learned solved: ',only_learned)
            print('Only Z3 solved: ',only_z3)
            print('-> Speedup (number of operations):')
            if len(speedups) > 0:
                speedups.sort()
                print('Average speedup: ',np.mean(speedups))
                for p in PER:
                    print('Percentile ',p,': ',speedups[int(p*len(speedups))])
            print('-> Speedup (wall clock time):')
            if len(speedups_real) > 0:
                speedups_real.sort()
                print('Average speedup: ',np.mean(speedups_real))
                for p in PER:
                    print('Percentile ',p,': ',speedups_real[int(p*len(speedups_real))])

if __name__ == '__main__':
    main()
