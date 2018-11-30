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

import re
import subprocess
import tempfile
import threading
import time


PATTERN_1 = re.compile(r"(\d+)\s\[label = ([A-Z_]+),hash = (\d+)]")
PATTERN_2 = re.compile(r"\s+(\d+)\s+\-\>\s+(\d+)")


class GoalRunnerThread(threading.Thread):
    """ Runner which executes a single tactic on a single goal. """

    def __init__(self, goal_input, goal_output, tactic, timeout):
        threading.Thread.__init__(self)
        self.goal_input = goal_input
        self.goal_output = goal_output
        self.tactic = str(tactic)
        self.timeout = timeout

    def run(self):
        self.time_before = time.time()

        use_cpp = True

        if not use_cpp:
            prog = ['python', 'utils/goal_runner.py']
        else:
            prog = ['./cpp/goal_runner']

        args = prog + [
            self.tactic,
            self.goal_input,
            self.goal_output]

        self.p = subprocess.Popen(args, stdout=subprocess.PIPE)
        self.p.wait()
        self.time_after = time.time()

    def Run(self):
        # print('GoalRunnerThread: (' + str(self.tactic) + "), " + self.goal_input + " -> " + self.goal_output)
        self.start()
        self.join(self.timeout)

        if self.is_alive():
            try:
                self.p.terminate()
                self.join()
            except OSError:
                pass
            return False, 0, None, None, None, 0, []

        out, err = self.p.communicate()
        if out[:-1].decode("utf-8").strip() == '-1':
            # exception thrown in goal runner
            return False, 0, None, None, None, 0, []

        lines = out[:-1].decode("utf-8").split('\n')

        try:
            res, rlimit, in_hash, out_hash, runtime = lines[0].split(' ')
            rlimit = int(rlimit)
            runtime = float(runtime)
            assert rlimit >= 0

            probes = list(map(float, lines[1].split(' ')))
        except ValueError:
            return False, 0, None, None, None, 0, []

        return True, rlimit, res, in_hash, out_hash, runtime, probes

    
class RunnerThread(threading.Thread):
    """ Runner which executes a strategy on a given SMT2 formula. """

    def __init__(self, strategy, smt_file, timeout):
        threading.Thread.__init__(self)
        self.strategy = strategy
        self.smt_file = smt_file
        self.timeout = timeout

    def run(self):
        args = ['python',
                'utils/runner.py',
                self.strategy,
                self.smt_file]

        self.p = subprocess.Popen(args, stdout=subprocess.PIPE)
        self.p.wait()

    def Run(self):
        self.start()
        self.join(self.timeout)

        if self.is_alive():
            try:
                self.p.terminate()
                self.join()
            except OSError:
                pass
            return 'unknown', 0

        try:
            out, err = self.p.communicate()
            out = str(out, 'utf-8')
            res, rlimit = out[:-1].split(' ')
            rlimit = int(rlimit)
            return res, rlimit
        except ValueError:
            return 'unknown', 0
