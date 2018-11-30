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

from __future__ import print_function
import sys
import os
import time
import z3
from fastsmt.language.objects import *

assert len(sys.argv) >= 3, 'Not enough arguments provided'

def get_rlimit():
    tmp = z3.Solver()
    stats = tmp.statistics()
    for i in range(len(stats)):
        if stats[i][0] == 'rlimit count':
            return stats[i][1]
    return 0

# print 'Runner...'
# print sys.argv[1]

if sys.argv[1] == 'default':
    solver = z3.Solver()
else:
    solver = from_string(sys.argv[1]).tactic.solver()
    
smt_file = sys.argv[2]

f = z3.parse_smt2_file(smt_file)
solver.add(f)

rlimit_before = get_rlimit()
res = solver.check()
rlimit_after = get_rlimit()

print('{} {}'.format(res, rlimit_after - rlimit_before))


