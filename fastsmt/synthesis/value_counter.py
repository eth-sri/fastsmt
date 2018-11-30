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

import os
import pickle


class BinaryValue:

    def __init__(self):
        self.success = 0
        self.total = 0

    def Add(self, value):
        if value:
            self.success += 1
        self.total += 1

    def __str__(self):
        return "success: %d, total: %d" % (self.success, self.total)

    def __repr__(self):
        return str(self)


class BinaryValueCounter:

    def __init__(self):
        self.values = {}

    def Add(self, key, value):
        h = str(key)
        if h not in self.values:
            self.values[h] = BinaryValue()

        self.values[h].Add(value)
        # print('Add key %s hash(%d)' % (str(key), h))

    def DebugProb(self, key):
        print('\t: ' + str(key))
        h = str(key)
        if h not in self.values:
            print('\t\tprob: 1.0, key not found.')
            return

        value = self.values[h]
        print('\t\tvalid %d + 1 / total %d + 1 = %f' % (value.success, value.total, (value.success + 1) / (value.total + 1)))

    def Prob(self, key):
        h = str(key)
        if h not in self.values:
            # print('key %s not found. hash(%d)' % (str(key), h))
            return 1.0

        value = self.values[h]
        # print('key %s, value %s' % (key, str(value)))
        return (value.success + 1) / (value.total + 1)

    def SaveModel(self, file):
        print('Save ValueCounter with %d values' % (len(self.values)))
        with open(file, "wb") as f:
            pickle.dump(self.values, f)

    def LoadModel(self, file):
        if not os.path.isfile(file):
            return

        with open(file, "rb") as f:
            self.values = pickle.load(f)
        print('Load ValueCounter with %d values' % (len(self.values)))
