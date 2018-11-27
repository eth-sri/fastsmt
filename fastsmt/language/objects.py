import z3

ALL_TACTICS = [
    'simplify',
    'smt',
    'bit-blast',
    'bv1-blast',
    'solve-eqs',
    'aig',
    'qfnra-nlsat',
    'sat',
    'max-bv-sharing',
    'reduce-bv-size',
    'purify-arith',
    'propagate-values',
    'elim-uncnstr',
    'ackermannize_bv',
    'skip',
]

ALL_PROBES = [
    'num-consts',
    'num-exprs',
    'size',
    'depth',
    'ackr-bound-probe',
    'is-qfbv-eq',
    'arith-max-deg',
    'arith-avg-deg',
    'arith-max-bw',
    'arith-avg-bw',
    'is-unbounded',
    'is-pb',
    # 'is-qfbv',
]

ALLOWED_PARAMS = {}

def make_strategy(tactics):
    """ Given list of tactics, combines them using AndThen if there is more than one tactic. """
    if len(tactics) == 1:
        return Tactic(tactics[0]) if isinstance(tactics[0], str) else tactics[0]
    else:
        return AndThen(*tactics)

def shorten_strategy(strategy):
    """ Pop tactic from the end of strategy and return tuple consisting of remaining strategy and single tactic. """
    tactics = get_tactics(str(strategy))
    if len(tactics) == 1:
        return None, tactics[0]
    tactic = tactics.pop()
    return make_strategy(tactics), tactic

def head_strategy(strategy):
    """ Pop tactic from the beginning of strategy and return tuple consisting of single tactic and remaining strategy. """
    tactics = get_tactics(str(strategy))
    if len(tactics) == 1:
        return tactics[0], None
    tactic = tactics.pop(0)
    return tactic, make_strategy(tactics)

def get_strategy_suffix(full_strat, prefix_strat):
    """ Returns suffix strategy which completes prefix strategy to the full strategy. """
    return get_tactics(full_strat)[len(get_tactics(prefix_strat)):]

class Tactic:
    """ Wrapper class around Z3 Tactic object. """
    
    def __init__(self, s):
        """ Initializes object of type Tactic.

        :param s: name of the tactic
        """
        assert isinstance(s, str)
        self.s = s
        self.tactic = z3.Tactic(s)

    def __str__(self):
        return 'Tactic({})'.format(self.s)

    def compact_str(self):
        """ Returns compact string representation. """
        return str(self.s)

    def to_smt2(self):
        """ Returns Tactic in SMT2 format. """
        return self.s

class AndThen:
    """ Wrapper class around Z3 AndThen object. """

    def __init__(self, *args):
        """ Initializes object of type AndThen.

        :param args: list of Tactic objects which make up AndThen object
        """
        self.v = [Tactic(x) if isinstance(x, str) else x for x in args]
        self.tactic = z3.AndThen(*[x.tactic for x in self.v])

    def __str__(self):
        return 'AndThen({})'.format(','.join(map(str, self.v)))      

    def __eq__(self, other):
        return str(self) == str(other)

    def to_smt2(self):
        """ Returns AndThen object in SMT2 format. """
        return '(then ' + ' '.join([t.to_smt2() for t in self.v]) + ')'

class OrElse:

    def __init__(self, *args):
        self.v = [Tactic(x) if isinstance(x, str) else x for x in args]
        self.tactic = z3.OrElse(*[x.tactic for x in self.v])

    def __str__(self):
        return 'OrElse({})'.format(','.join(map(str, self.v)))
        
    def erase(self, i):
        assert i >= 0 and i < len(self.v)
        self.v = self.v[:i] + self.v[i+1:]
        self.tactic = z3.OrElse(*[x.tactic for x in self.v])

    def insert(self, i, x):
        if isinstance(x, str):
            x = Tactic(x)
        self.v = self.v[:i] + [x] + self.v[i+1:]
        self.tactic = z3.OrElse(*[x.tactic for x in self.v])

    def to_smt2(self):
        assert False

class Probe:
    """ Wrapper class around Z3 Probe object. """
    
    def __init__(self, s):
        """ Initializes object of Probe.

        :param s: name of the probe
        """
        assert isinstance(s, str)

        self.s = s
        self.probe = z3.Probe(s)

    def __call__(self, g):
        return self.probe(g)

    def __str__(self):
        return 'Probe({})'.format(self.s)

class With:
    """ Wrapper class around Z3 With object. """
    
    def __init__(self, s, params):
        assert isinstance(s, str)

        self.s = s
        self.params = params
        self.tactic = z3.With(s, **params)

    def __str__(self):
        param_str = ';'.join(sorted(['{}={}'.format(x, self.params[x]) for x in self.params]))
        return 'With({};{})'.format(self.s, param_str)

    def compact_str(self):
        params = [int(self.params[x]) for x in self.params]
        return self.s + '(' + ','.join(list(map(str,params))) + ')'

    def to_smt2(self):
        """ Returns With object in SMT2 format. """
        param_str = []
        for x in self.params:
            if isinstance(self.params[x], bool):
                eval = 'true' if self.params[x] else 'false'
            else:
                eval = str(self.params[x])
            param_str.append(':%s %s' % (x, eval))

        return '(using-params %s %s)' % (self.s, ' '.join(param_str))


def get_tactics(s):
    """ Given strategy in string format, decomposes it into list of Tactic objects.

    :param s: strategy in the string format
    :return: list of Tactic objects that strategy consists of
    """
    s = str(s)
    if s in ALL_TACTICS or s == 'skip':
        return [Tactic(s)]
    elif s[:7] == 'Tactic(' and s[-1] == ')':
        return [Tactic(s[7:-1])]
    elif s[:8] == 'AndThen(' and s[-1] == ')':
        tokens = s[8:-1].split(',')
        tactics = list(map(from_string, tokens))
        return tactics
    elif s[:5] == 'With(' and s[-1] == ')':
        tokens = s[5:-1].split(';')
        tactic = tokens[0]

        params = {}
        for token in tokens[1:]:
            x, val = token.split('=')
            if val == 'True':
                params[x] = True
            elif val == 'False':
                params[x] = False
            elif val.isdigit():
                params[x] = int(val)
            else:
                assert False, 'param {} = {} invalid'.format(x, val)

        return [With(tactic, params)]
    assert False, 'string {} is invalid strategy'.format(s)

def from_string(s):
    """ Given strategy in string format, returns one of wrapper classes which strategy is represented by.

    :param s: strategy in the string format
    :param all_tactics: list of tactics
    :return: object of one of the wrapper classes which represents the strategy
    """
    if s[:7] == 'Tactic(' and s[-1] == ')':
        return Tactic(s[7:-1])
    elif s[:8] == 'AndThen(' and s[-1] == ')':
        tokens = s[8:-1].split(',')
        tactics = list(map(from_string, tokens))
        return AndThen(*tactics)
    elif s[:5] == 'With(' and s[-1] == ')':
        tokens = s[5:-1].split(';')
        tactic = tokens[0]
        
        params = {}
        for token in tokens[1:]:
            x, val = token.split('=')
            if val == 'True':
                params[x] = True
            elif val == 'False':
                params[x] = False
            elif val.isdigit():
                params[x] = int(val)
            else:
                assert False, 'param {} = {} invalid'.format(x,val)
                
        return With(tactic, params)
    else:
        return Tactic(s)
    assert False, 'string {} is invalid strategy'.format(s)
        

