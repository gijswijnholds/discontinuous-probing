from .preprocessing import CompactSample, SpanDataset
from .data_reader import abstree_to_rules
from itertools import groupby
from operator import eq
from typing import Any
from math import sqrt


def sort_by_acc(xs): return sorted(xs.items(), key=lambda x: -x[1][-1])
def sort_by_key(xs): return sorted(xs.items(), key=lambda x: x[0])


def analysis(test_data: SpanDataset, predictions: list[list[int]]):
    gbd, gbv, gbn = verb_depth_acc(predictions, cs := [d.compact for d in test_data])
    gbt = tree_acc(predictions, cs)
    gbr = rule_acc(predictions, cs)
    correct, total, _ = list(zip(*gbd.values()))
    return {'total': (sum(correct), sum(total), sum(correct) / sum(total)),
            'acc_by_depth': gbd,
            'acc_by_verb': gbv,
            'acc_by_#nouns': gbn,
            'baseline': baseline([(len(c.n_spans), len(c.labels)) for c in cs]),
            'acc_by_tree': gbt,
            'acc_by_rule': gbr}


def verb_depth_acc(pss: list[list[int]], samples: list[CompactSample]) \
        -> tuple[dict[Any, tuple[int, int, float]], ...]:
    all_preds = [
        (' '.join([s for idx, s in enumerate(sample.sentence) if sample.v_spans[i][idx] == 1]),
         p == sample.labels[i], sample.depth, len(sample.n_spans))
        for sample, ps in zip(samples, pss) for i, p in enumerate(ps)]

    def gb(key: int):
        return [(k, [v[1] for v in vs]) for k, vs in groupby(sorted(all_preds, key=lambda x: x[key]), lambda x: x[key])]

    return ({k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gb(-2) if len(vs)},
            {k: (c := sum(vs), ln := len(vs), c / ln) for k, vs in gb(0) if len(vs)},
            {k: (c := sum(vs), ln := len(vs), c / ln) for k, vs in gb(-1) if len(vs)})


def tree_acc(pss: list[list[int]], samples: list[CompactSample]):
    all_preds = [(eq(ps, sample.labels), sample.abstree) for ps, sample in zip(pss, samples)]

    gbt = [(k, [v[0] for v in vs]) for k, vs in groupby(sorted(all_preds, key=lambda x: str(x[-1])),
                                                        key=lambda x: x[-1])]
    return {k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gbt}


def rule_acc(pss: list[list[int]], samples: list[CompactSample]):
    all_preds = [(eq(ps, sample.labels), rule) for ps, sample in zip(pss, samples)
                 for rule in abstree_to_rules(sample.abstree)]
    gbr = [(k, [v[0] for v in vs]) for k, vs in groupby(sorted(all_preds, key=lambda x: x[-1]), key=lambda x: x[-1])]
    return {k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gbr}


def baseline(nss: list[tuple[int, int]]) -> float:
    chances = [1/ns for ns, vs in nss for _ in range(vs)]
    return sum(chances) / len(chances)


def agg_torch_seeds(results: dict[str, dict[str, Any]]):
    def merge(xs):
        if isinstance(xs[0], (float, int)):
            return (mu := sum(xs)/len(xs)), sqrt(sum([(x-mu) ** 2 for x in xs])/len(xs))
        if isinstance(xs[0], tuple):
            return tuple(map(merge, zip(*xs)))
        if isinstance(xs[0], dict):
            return {k: merge([x[k] for x in xs]) for k in xs[0].keys()}
        raise TypeError(f'Cannot merge {type(xs[0])}')

    seeds = [results[s] for s in results if s != 'dataset']
    metrics = [k for k in seeds[0].keys()]
    return {metric: merge([seed[metric] for seed in seeds]) for metric in metrics}

#
# def aggregate(file: str):
#     with open(file, 'rb') as in_file:
#         data = pickle.load(in_file)
#     agg_results = defaultdict(list)
#     for experiment in data:
#         exp_name, data_seed = experiment.split(':')[1].split('_')
#         agg_results[(exp_name, int(data_seed))] = [analysis(data[experiment][repeat])
#                                                    for repeat in data[experiment]]
#     return {k: merge_results(agg_results[k]) for k in agg_results}
#
#
# def show_results(results: dict):
#     names, data_seeds = list(zip(*results.keys()))
#     names = ['vocab', 'depth', 'rule', 'all']
#     data_seeds = sorted(list(set(data_seeds)))
#     for name in names:
#         print('=' * 64)
#         print(name.upper())
#         print('=' * 64)
#         d_accs = [d_acc for dseed in data_seeds for d_acc in results[(name, dseed)]['acc_by_depth'].items()]
#         n_accs = [n_acc for dseed in data_seeds for n_acc in results[(name, dseed)]['acc_by_num_nouns'].items()]
#         agg_acc_by_depth = agg_result(d_accs)
#         agg_acc_by_num_nouns = agg_result(n_accs)
#         agg_baseline = avg([results[(name, data_seed)]['baseline'] for data_seed in data_seeds]),
#         correct, total, _ = tuple(map(avg, zip(*[results[(name, data_seed)]['total'] for data_seed in data_seeds])))
#         best_epochs = [best_epoch for dseed in data_seeds for best_epoch in results[(name, dseed)]['best_epochs']]
#         print('Accuracy by depth:')
#         print('\n'.join([f'{d}:\t{c:1.0f}\t{t:1.0f}\t({a:0.2f})' for d, (c, t, a) in agg_acc_by_depth.items()]))
#         print('=' * 64)
#         print('Accuracy by number of nouns:')
#         print('\n'.join([f'{d}:\t{c:1.0f}\t{t:1.0f}\t({a:0.2f})' for d, (c, t, a) in agg_acc_by_num_nouns.items()]))
#         print('=' * 64)
#         print('Best epochs:')
#         print(best_epochs)
#         print('=' * 64)
#         print('Aggregated baseline:')
#         print(round(agg_baseline[0], 2))
#         print('Average accuracy:')
#         print(round(correct/total, 2))
#     print('=' * 64)
#     print('=' * 64)
#     print('=' * 64)
#     print('Results for verbs (exp, seed)')
#     for name in names:
#         for data_seed in data_seeds:
#             print('=' * 64)
#             print(f'{name.upper()}\t(Seed: {data_seed})')
#             print('=' * 64)
#             verb_results = results[(name, data_seed)]['acc_by_verb']
#             print('\n'.join([f'{d[0]}\t{round(d[1][2], 2)}' for d in sorted(verb_results.items(),
#                                                                             key=lambda x: x[1][-1])]))
#     print('=' * 64)
#     print('=' * 28 + ' THE END ' + '=' * 27)
#     print('=' * 64)
#
#
