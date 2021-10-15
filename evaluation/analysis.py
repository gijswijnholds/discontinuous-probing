from .preprocessing import CompactSample, SpanDataset
from itertools import groupby
from typing import Any
import pickle
from collections import defaultdict


def analysis(experiment):
    epochs = {k: v for k, v in experiment.items() if isinstance(k, int)}
    test_data: SpanDataset = experiment['test_data']
    best_epoch = max(epochs, key=lambda k: epochs[k]['val_acc'])
    predictions = epochs[best_epoch]['test_preds']
    gbd, gbv, gbn = verb_depth_acc(predictions, [d.compact for d in test_data])
    correct, total, _ = list(zip(*gbd.values()))
    def sort_by_acc(xs): return sorted(xs.items(), key=lambda x: -x[1][-1])
    def sort_by_key(xs): return sorted(xs.items(), key=lambda x: x[0])
    results = dict()
    results['acc_by_depth'] = sort_by_key(gbd)
    results['acc_by_verb'] = sort_by_acc(gbv)
    results['acc_by_num_nouns'] = sort_by_key(gbn)
    results['baseline'] = baseline(predictions)
    results['total'] = (sum(correct), sum(total), sum(correct)/sum(total))
    results['best_epoch'] = best_epoch
    return results


def verb_depth_acc(pss: list[list[int]], samples: list[CompactSample]) \
        -> tuple[dict[Any, tuple[int, int, float]], ...]:
    all_preds = [(' '.join([sample.sentence[idx] for idx in sample.v_spans[i] if idx == 1]),
                  p == sample.labels[i], sample.depth, len(sample.n_spans))
                 for sample, ps in zip(samples, pss) for i, p in enumerate(ps)]
    gbv = [(k, [v[1] for v in vs]) for k, vs in groupby(sorted(all_preds, key=lambda x: x[0]), lambda x: x[0])]
    gbd = [(k, [v[1] for v in vs]) for k, vs in groupby(sorted(all_preds, key=lambda x: x[-2]), lambda x: x[-2])]
    gbn = [(k, [v[1] for v in vs]) for k, vs in groupby(sorted(all_preds, key=lambda x: x[-1]), lambda x: x[-1])]
    return ({k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gbd if len(vs)},
            {k: (c := sum(vs), ln := len(vs), c / ln) for k, vs in gbv if len(vs)},
            {k: (c := sum(vs), ln := len(vs), c / ln) for k, vs in gbn if len(vs)})


def baseline(pss: list[list[int]]) -> float:
    chance = [1/len(ps) for ps in pss]
    return sum(chance) / len(chance)


def avg(ln):
    return sum(ln) / len(ln)


def get_group_agg(results: list, field: str):
    sort_results = sorted([d_res for res in results for d_res in res[field]], key=lambda r: r[0])
    grouped_results = {k: [v[1] for v in vs] for k, vs in groupby(sort_results, key=lambda r: r[0])}
    avg_results = {k: tuple(map(avg, zip(*vs))) for k, vs in grouped_results.items()}
    return avg_results


def merge_results(all_results: list[dict]) -> dict:
    merged_results = dict()

    merged_results['acc_by_depth'] = get_group_agg(all_results, 'acc_by_depth')
    merged_results['acc_by_verb'] = get_group_agg(all_results, 'acc_by_verb')
    merged_results['acc_by_num_nouns'] = get_group_agg(all_results, 'acc_by_num_nouns')
    merged_results['baseline'] = avg([res['baseline'] for res in all_results])
    merged_results['total'] = tuple(map(avg, zip(*[res['total'] for res in all_results])))
    merged_results['best_epochs'] = [res['best_epoch'] for res in all_results]
    return merged_results


def aggregate(file: str):
    with open(file, 'rb') as in_file:
        data = pickle.load(in_file)
    agg_results = defaultdict(list)
    for experiment in data:
        exp_name, data_seed = experiment.split(':')[1].split('_')
        agg_results[(exp_name, int(data_seed))] = [analysis(data[experiment][repeat])
                                                   for repeat in data[experiment]]
    return {k: merge_results(agg_results[k]) for k in agg_results}


def show_results(results: dict):
    names, data_seeds = list(zip(*results.keys()))
    names, data_seeds = set(names), set(data_seeds)
    for name in names:
        print(name)
        print('=' * 64)

        d_accs = [{'acc_by_depth': results[(name, dseed)]['acc_by_depth']} for dseed in data_seeds]
        n_accs = [{'acc_by_num_nouns': results[(name, dseed)]['acc_by_num_nouns']} for dseed in data_seeds]
        agg_acc_by_depth = get_group_agg(d_accs, 'acc_by_depth')
        agg_acc_by_num_nouns = get_group_agg(n_accs, 'acc_by_num_nouns')
        agg_baseline = avg([results[(name, data_seed)]['baseline'] for data_seed in data_seeds]),
        correct, total, _ = tuple(map(avg, zip(*[results[(name, data_seed)]['total'] for data_seed in data_seeds])))
        best_epochs = [best_epoch for dseed in data_seeds for best_epoch in results[(name, dseed)]['best_epochs']]
        print('=' * 64)
        print('Accuracy by depth:')
        print('\n'.join([f'{d}:\t{c}\t{t}\t({a:0.2f})' for d, (c, t, a) in agg_acc_by_depth.items()]))
        print('=' * 64)
        print('Accuracy by number of nouns:')
        print('\n'.join([f'{d}:\t{c}\t{t}\t({a:0.2f})' for d, (c, t, a) in agg_acc_by_num_nouns.items()]))
        print('=' * 64)
        print('Best epochs:')
        print(best_epochs)
        print('Aggregated baseline:')
        print(agg_baseline)
        print('Average accuracy:')
        print({correct/total})
