import argparse
import json
import os
import seaborn as sns
from fastsmt.language.objects import get_tactics
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)

sns.set_style("darkgrid")
sns.set(font_scale=1.5)
sns.set_palette(sns.color_palette("deep", 5))

NUM_ITERS = 100

NICE_NAME = {
    'sage2_bilinear': 'Bilinear',
    'sage2_apprentice': 'NN',
    'sage2_random': 'Random',
    'sage2_bfs': 'BFS',
    'sage2_evo': 'Evolutionary',
}


def to_nice(model):
    # if model[0] in NICE_NAME:
    #     return NICE_NAME[model[0]]
    if model[0] in NICE_NAME:
        return NICE_NAME[model[0]] + ':' + str(model[1])
    return model[0] + '_' + str(model[1])


class Stats:

    def __init__(self, legend, max_distance, model_names):
        self.legend = legend
        self.stats = {}
        self.model_names = model_names
        self.rh = {}
        self.solved = {}

        for i in range(1, max_distance + 1):
            self.stats[i] = {}
            for model_name in model_names:
                self.solved[model_name] = 0
                self.stats[i][model_name] = {'total': 0, 'success': 0, 'rank': []}
                self.rh[model_name] = []

    def add_solved(self, model_name):
        self.solved[model_name] += 1

    def add(self, distance, it, model_name):
        if it != -1:
            self.stats[distance][model_name]['success'] += 1
            self.stats[distance][model_name]['rank'].append(it)
        self.stats[distance][model_name]['total'] += 1

    def append_relative_history(self, model, relative_history):
        if len(self.rh[model]) > 0:
            assert len(self.rh[model][0]) == len(relative_history)
        self.rh[model].append(list(map(float, relative_history)))

    def plot(self):
        plt.figure(figsize=(10, 5))

        lines = []
        for model in self.rh:
            self.rh[model] = np.array(self.rh[model])
            mean = np.mean(self.rh[model], axis=0)
            x = [it for it in range(1, mean.shape[0]+1)]
            l, = plt.plot(x, mean, label=to_nice(model), linewidth=2.6,
                          linestyle='dashed' if model == 'Default' else 'solid')
            lines.append(l)

        if self.legend:
            l2 = lines
            l2[0], l2[-1] = l2[-1], l2[0]
            plt.legend(handles=l2, fontsize=16)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
                    
        plt.xlabel('Number of sampled strategies', fontsize=24)
        plt.ylabel('Performance', fontsize=24)
        plt.gcf().subplots_adjust(bottom=0.18)

        sns.set_context("paper", rc={"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 5})

        plt.ylim([0, 1])
        plt.savefig('search_strategies.pdf', format='pdf')
        plt.show()

    def print(self):
        print('===================')
        for distance in sorted(self.stats.keys()):
            print('Distance: ', distance)
            stat = self.stats[distance]
            for model in self.model_names:
                print('\t', model, 'total: ', stat[model]['total'],
                      'success: ', stat[model]['success'],
                      ', avg rank: ', np.average(stat[model]['rank']))
        print('===================')
        print(self.solved)


def analyze_tc(tc, stats):
    best_rlimit = -1
    best_strategy = None

    until_best = {}

    ok = True

    for run in tc['runs']:
        best_run_rlimit = run['history'][-1]
        if best_rlimit == -1 or (best_run_rlimit != -1 and best_run_rlimit < best_rlimit):
            best_rlimit = best_run_rlimit
            best_strategy = run['best_strategy']

    if not ok or best_rlimit == -1:
        return

    # for each model, calculate number of iterations to reach best_rlimit
    for run in tc['runs']:
        until_best[run['model_name']] = -1

        for it, rlimit in enumerate(run['history']):
            if rlimit == best_rlimit:
                until_best[run['model_name']] = it + 1
                break

        for rlimit in run['history']:
            if rlimit != -1:
                stats.add_solved(run['model_name'])
                break

        relative_history = [best_rlimit / float(x) if x > 0 else 0 for x in run['history']]
        stats.append_relative_history(run['model_name'], relative_history)


    if best_strategy is not None:
        best_tactics = list(map(str, get_tactics(best_strategy)))
        best_len = len(best_tactics)
    else:
        best_len = 27

    for model in until_best:
        stats.add(best_len, until_best[model], model)

def solved(res):
    return (res == 'sat' or res == 'unsat')

def main():
    parser = argparse.ArgumentParser(description='Evaluate results of synthesis')
    parser.add_argument('--eval_dir', type=str, help='Directory with results')
    parser.add_argument('--folder', type=str, default='train')
    parser.add_argument('--legend', action='store_true', help='Show legend.')
    parser.add_argument('--models', nargs='+', help='Models which should be plotted', required=False)
    args = parser.parse_args()

    print(args.models)
    
    models = []
    for model in args.models:
        tokens = model.split(':')
        models.append((tokens[0], int(tokens[1])))
    
    stats = Stats(args.legend, 15, models)
    data = {}

    for model, version in models:
        model_dir = os.path.join(args.eval_dir, model, args.folder, str(version))
        for root, directories, filenames in os.walk(model_dir):
            for file in filenames:
                if not file.endswith('.log'):
                    continue
                
                if file not in data:
                    data[file] = {'runs': []}
                    
                with open(os.path.join(model_dir, file), 'r') as f:
                    history = []
                    best_candidate = None

                    for line in f:
                        if line[:-1] != 'None':
                            candidate = json.loads(line[:-1])

                            if solved(candidate['res']) and ((best_candidate is None) or candidate['rlimit'] < best_candidate['rlimit']):
                                best_candidate = candidate

                        if best_candidate is None:
                            history.append(-1)
                        else:
                            history.append(best_candidate['rlimit'])

                last = history[-1]
                while len(history) < NUM_ITERS:
                    history.append(last)

                if len(history) > NUM_ITERS:
                    history = history[:NUM_ITERS]
                    
                data[file]['runs'].append(
                    {
                        'history': history,
                        'model_name': (model, version),
                        'best_strategy': best_candidate['strat'] if best_candidate is not None else None
                    })

    for file in data:
        if len(data[file]['runs']) != len(models):
            print('skip ',file)
            assert False
            continue
        analyze_tc(data[file], stats)

    stats.print()
    stats.plot()

if __name__ == '__main__':
    main()




