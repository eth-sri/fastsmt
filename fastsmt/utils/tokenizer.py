"""
Utility class for parsing formula into tokens or
creating bag of words representation.
"""
import re

BV_THEORY = [
    'bvadd', 'bvsub', 'bvneg', 'bvmul', 'bvurem', 'bvsrem', 'bvsmod',
    'bvshl', 'bvlshr', 'bvashr', 'bvor', 'bvand', 'bvnand', 'bvnor',
    'bvxnor', 'bvule', 'bvult', 'bvugt', 'bvuge', 'bvsle', 'bvslt',
    'bvsge', 'bvsgt', 'bvudiv', 'extract', 'bvudiv_i', 'bvnot',
]

ST_TOKENS = [
    '=', '<', '>', '==', '>=', '<=', '=>', '+', '-', '*', '/',
    'true', 'false', 'not', 'and', 'or', 'xor',
    'zero_extend', 'sign_extend', 'concat', 'let', '_', 'ite',
    'exists', 'forall', 'assert', 'declare-fun', 
    'Int', 'Bool', 'BitVec',
]

ALL_TOKENS = ["UNK"] + ST_TOKENS + BV_THEORY

class Tokenizer(object):

    def __init__(self):
        self.token_idx = {}
        for i, token in enumerate(ALL_TOKENS):
            self.token_idx[token] = i


    def tokenize(self, filename):
        with open(filename, 'r') as f:
            txt = f.read()
                
            digit = re.compile("\d+")
                
            txt = re.sub('[\(\)\n]', ' ', txt) 
            txt = re.sub('[ ]+', ' ', txt)
                
            tokens = txt.split(' ')
                
            idx = []
            for token in tokens:
                if token not in self.token_idx:
                    idx.append(0)
                else:
                    idx.append(self.token_idx[token])
            return idx
        return None

    def bag_of_words(self, idx):
        ret = [0 for _ in ALL_TOKENS]
        for x in idx:
            ret[x] += 1
        return ret
