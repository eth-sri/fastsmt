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


