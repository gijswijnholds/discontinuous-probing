from .preprocessing import CompactSample, SpanDataset
from itertools import groupby
from typing import Any


def analysis(experiment, verbose: bool = True):
    def show(whatever: str): print(whatever) if verbose else None
    epochs = {k: v for k, v in experiment.items() if isinstance(k, int)}
    test_data: SpanDataset = experiment['test_data']
    best_epoch = max(epochs, key=lambda k: epochs[k]['val_acc'])
    predictions = epochs[best_epoch]['test_preds']
    gbd, gbv, gbn = verb_depth_acc(predictions, [d.compact for d in test_data])
    correct, total, _ = list(zip(*gbd.values()))
    def sort_by_acc(xs): return sorted(xs.items(), key=lambda x: -x[1][-1])
    def sort_by_key(xs): return sorted(xs.items(), key=lambda x: x[0])
    print('Accuracy by depth:')
    print('\n'.join([f'{d}:\t{c}\t{t}\t({a:0.2f})' for d, (c, t, a) in sort_by_key(gbd)]))
    print('=' * 64)
    print('Acccuracy by verb:')
    print('\n'.join([f'{d}:\t{c}\t{t}\t({a:0.2f})' for d, (c, t, a) in sort_by_acc(gbv)]))
    print('=' * 64)
    print('Acccuracy by number of nouns:')
    print('\n'.join([f'{d}:\t{c}\t{t}\t({a:0.2f})' for d, (c, t, a) in sort_by_key(gbn)]))
    print('=' * 64)
    print(f'Baseline accuracy: ({round(baseline(predictions), 2)})')
    print('=' * 64)
    print(f'Total accuracy:\t{sum(correct)}\t{sum(total)}\t({round(sum(correct)/sum(total), 2)})')


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


def aggregate(file):
