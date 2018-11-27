import copy
import random
from fastsmt.synthesis.search_strategies import *

class EvoSearch(SearchStrategy):


    def __init__(self, tester, config, model):
        assert isinstance(tester, BenchmarkGoalTester)
        super(EvoSearch, self).__init__(tester)

        self.config = config
        self.model = model
        tactics_config = config["tactics_config"]
        self.strategy_enum = StrategyEnumerator(**tactics_config)

        self.stats = {
            'num_pruned': 0,
            'num_redundant': 0,
            'num_success': 0,
            'num_solved': 0,
            'num_timeout': 0,
        }

    def clear(self):
        self.valid = False
        self.skip_tests = None

        # all strategies that might be scored in the future
        self.unscored_strategies = {smt_instance: [] for smt_instance in self.smt_instances}
        self.scored_strategies = {smt_instance: [] for smt_instance in self.smt_instances}
        self.best_strategy = {smt_instance: None for smt_instance in self.smt_instances}
        self.population = {smt_instance:[] for smt_instance in self.smt_instances}

        # selected strategies to be evaluated at current step
        self.candidates = []

        # unique sequence count
        self.counter = itertools.count()

        self.stats = {
            'num_pruned': 0,
            'num_redundant': 0,
            'num_success': 0,
            'num_solved': 0,
            'num_timeout': 0,
        }

    def restart(self, smt_instances, valid):
        self.smt_instances = smt_instances
        self.per_formula_strategies = {None: FormulaEqClass(None)}
        self.clear()
        self.valid = valid

    def get_stats(self):
        return str(self.stats) + ', scored strategies: %d' % (len(self.scored_strategies))

    def select_candidates(self, desired_size, smt_instance):
        """ Selects best candidate strategies that should be tested.

        :param desired_size: number of candidates to select
        :param smt_instance: smt instance for which synthesis is performed
        :return: number of candidates selected
        """
        for i, entry in enumerate(self.unscored_strategies[smt_instance]):
            candidate_tactics, strategy = entry

            if strategy is None:
                parent = Strategy(self.skip_tests[smt_instance].strat)
                parent.add_benchmark_test(self.skip_tests[smt_instance])
            else:
                parent = strategy

            child_score = self.model.score_strategy(strategy=candidate_tactics, parent=parent)

            heapq.heappush(self.scored_strategies[smt_instance],
                           (-1 * child_score, next(self.counter), entry))

        self.unscored_strategies[smt_instance] = []

        candidates_added = 0
        while candidates_added < desired_size and self.scored_strategies[smt_instance]:
            priority, count, entry = heapq.heappop(self.scored_strategies[smt_instance])
            new_strategy = Strategy(make_strategy(entry[0]))
            new_strategy.score = -priority
            self.candidates.append((smt_instance, new_strategy))
            candidates_added += 1

        return candidates_added


    def init_population(self, desired_size, initial_candidate=None, smt_instances=None):
        """ Initializes population of strategies.

        :param desired_size: number of strategies to initialize with
        :param initial_candidate: initial candidate strategy, empty if None
        :param smt_instances: list of smt instances for which synthesis should be performed
        """
        TMP_DIR = 'tmp/fastsmt/search'
        if TMP_DIR is not None and not os.path.isdir(TMP_DIR):
            os.makedirs(TMP_DIR)
        if smt_instances is None:
            smt_instances = self.smt_instances

        if self.skip_tests is None:
            self.skip_tests = {
                smt_instance:
                    BenchmarkGoalTest(
                        smt_instance,
                        make_strategy(['skip']),
                        TMP_DIR,
                        timeout=10,
                        tester=self.tester)
                for smt_instance in self.smt_instances
            }
            self.tester.evaluate_parallel(self.skip_tests.values())

        self.model.reset()
        self.log.info('Population initialized!')

        for smt_instance in smt_instances:
            self.add_candidates_from_strategy(initial_candidate, smt_instance, True)
            self.select_candidates(desired_size, smt_instance)

    def add_candidates_from_strategy(self, strategy, smt_instance, add_all=False):
        tactics = [] if strategy is None else get_tactics(strategy.t)

        for i in range(1, len(tactics) - 1):
            candidate_tactics = tactics[:i] + tactics[i + 1:]
            if StrategyEnumerator.is_valid_strategy(candidate_tactics) and (
                    'max_strategy_size' not in self.config or len(candidate_tactics) <= self.config[
                'max_strategy_size']):
                self.unscored_strategies[smt_instance].append(
                    (candidate_tactics,
                     strategy if strategy and strategy.benchmarks else None))

        for i in range(0, len(tactics) + 1):
            if add_all:
                itset = self.strategy_enum.base_tactics
            else:
                itset = copy.deepcopy(self.strategy_enum.base_tactics)
                random.shuffle(itset)
                itset = itset[:4]
                
            for tactic in itset:
                #for tactic in self.strategy_enum.base_tactics:
                args = self.model.predict_arguments(tactic.s)
                with_tactic = self.strategy_enum.get_tactic_with_args(tactic.s, args)

                candidate_tactics = tactics[:i] + [with_tactic] + tactics[i:]
            
                if StrategyEnumerator.is_valid_strategy(candidate_tactics) and (
                        'max_strategy_size' not in self.config or len(candidate_tactics) <= self.config['max_strategy_size']):
                    self.unscored_strategies[smt_instance].append(
                        (candidate_tactics,
                         strategy if strategy and strategy.benchmarks else None))

    def extend_population(self, desired_size, smt_instance):
        return self.select_candidates(desired_size, smt_instance) > 0

    def Prune(self):
        new_candidates = []
        for strategy, parent in self.unscored_strategies:
            if parent and parent.pruned:
                continue
            if parent and self.best_strategy is not None and self.best_strategy.rlimit <= parent.rlimit:
                continue
            new_candidates.append((strategy, parent))
        # print('Pruned Candidates %d -> %d' % (len(self.unscored_strategies), len(new_candidates)))
        self.unscored_strategies = new_candidates

    def add_scored_strategy(self, candidate, status):
        smt_instance = candidate.benchmarks[0].file
        self.population[smt_instance].append((status, candidate))
        self.model.add_scored_strategy(candidate, status)
        self.tester.add_scored_strategy(candidate, status)
    
    def shrink_population(self, smt_instance):
        def cmpf(x):
            status, candidate = x
            rlimit = candidate.rlimit
            if status == ScoredCandidateStatus.SOLVED:
                return (-1, rlimit)
            elif status == ScoredCandidateStatus.SUCCESS:
                return (0, rlimit)
            else:
                return (1, rlimit)

        self.population[smt_instance].sort(key=cmpf)

        if len(self.population[smt_instance]) > self.config['evo_pop_size']:
            self.population[smt_instance] = self.population[smt_instance][:self.config['evo_pop_size']]

        # print('======== Population =========')
        # print('Total: ', len(self.population))
        # for status, candidate in self.population[smt_instance]:
        #     print(status, candidate)
        #     # assert status == ScoredCandidateStatus.SOLVED
        # print('=============================')

        self.unscored_strategies[smt_instance] = []
        for status, candidate in self.population[smt_instance]:
            self.add_candidates_from_strategy(candidate, smt_instance)


    def prune(self, smt_instance):
        """ Prunes all strategies which can not possibly lead to optimal strategy for solving smt instance. """
        new_candidates = []
        for strategy, parent in self.unscored_strategies[smt_instance]:
            if parent and parent.pruned:
                continue
            if parent and (self.best_strategy[smt_instance] is not None) and self.best_strategy[smt_instance].rlimit <= parent.rlimit:
                continue
            new_candidates.append((strategy, parent))
        self.unscored_strategies[smt_instance] = new_candidates

    def finished_evaluating(self, scored_candidates):
        """ Callback function called after evaluation of candidates is finished.

        :param scored_candidates: candidate strategies which were evaluated on smt instances
        """
        self.log.info('Finished evaluating %d candidates' % len(scored_candidates))
        for candidate in scored_candidates:
            smt_instance = candidate.benchmarks[0].file

            if candidate.all_solved() and (
                    self.best_strategy[smt_instance] is None or candidate.rlimit < self.best_strategy[smt_instance].rlimit):
                self.best_strategy[smt_instance] = candidate
                self.stats['num_solved'] += 1
                self.add_scored_strategy(candidate, ScoredCandidateStatus.SOLVED)
                self.prune(smt_instance)
                continue

            # print(candidate.get_goal_hashes())
            in_hashes, out_hashes = candidate.get_goal_hashes()
            if in_hashes == out_hashes:
                # print('PRUNED_EQ: ', str(candidate), candidate.get_goal_hashes())
                self.stats['num_pruned'] += 1
                candidate.pruned = True
                self.add_scored_strategy(candidate, ScoredCandidateStatus.PRUNED)
                continue
            if self.best_strategy[smt_instance] is not None and self.best_strategy[
                smt_instance].rlimit <= candidate.rlimit:
                # print('pruned rlimit: ' + str(candidate))
                self.stats['num_timeout'] += 1
                self.add_scored_strategy(candidate, ScoredCandidateStatus.TIMEOUT)
                continue

            if str(out_hashes) not in self.per_formula_strategies:
                self.per_formula_strategies[str(out_hashes)] = FormulaEqClass(candidate)
                #self.add_candidates_from_strategy(candidate, smt_instance)
                self.stats['num_success'] += 1
                self.add_scored_strategy(candidate, ScoredCandidateStatus.SUCCESS)
            else:
                formula = self.per_formula_strategies[str(out_hashes)]
                formula.count += 1

                if formula.candidate.rlimit > candidate.rlimit:
                    formula.candidate.pruned = True
                    formula.candidate = candidate
                    #self.add_candidates_from_strategy(candidate, smt_instance)
                    self.stats['num_success'] += 1
                    self.add_scored_strategy(candidate, ScoredCandidateStatus.SUCCESS)
                    self.prune(smt_instance)
                else:
                    self.stats['num_redundant'] += 1
                    candidate.pruned = True
                    self.add_scored_strategy(candidate, ScoredCandidateStatus.REDUNDANT)

        self.candidates = []

        t1 = time.time()
        for smt_instance in self.smt_instances:
            self.shrink_population(smt_instance)
        t2 = time.time()
        print('Shrinking done in ',t2-t1,' seconds')
