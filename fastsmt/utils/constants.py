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

PROBE_TO_ID = {
    'num-consts': 0,
    'num-exprs': 1,
    'size': 2,
    'depth': 3,
    'ackr-bound-probe': 4,
    'is-qfbv-eq': 5,
    'arith-max-deg': 6,
    'arith-avg-deg': 7,
    'arith-max-bw': 8,
    'arith-avg-bw': 9,
    'is-unbounded': 10,
    'is-pb': 11,
    # 'is-qfbv': 12,
}

TIMEOUT = 10000000000
