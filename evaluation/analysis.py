from .preprocessing import CompactSample, SpanDataset
from .data_reader import (LabeledTree, Rule, labeled_to_abstree, labtree_to_branches,
                          branch_to_rule, abstree_to_rules)
from itertools import groupby
from operator import eq
from typing import Any
from math import sqrt


def sort_by_acc(xs): return sorted(xs.items(), key=lambda x: -x[1][-1])
def sort_by_key(xs): return sorted(xs.items(), key=lambda x: x[0])
def mean(xs): return sum(xs) / len(xs)


def analysis(test_data: SpanDataset, predictions: list[list[int]]):
    gbd, gbv, gbn, gbv2 = verb_depth_acc(predictions, cs := [d.compact for d in test_data])
    gbr = rule_acc(predictions, cs)
    correct, total, _ = list(zip(*gbd.values()))
    gbt = tree_acc(predictions, cs)
    return {'total': (sum(correct), sum(total), sum(correct) / sum(total)),
            'acc_by_depth': gbd,
            'acc_by_verb': gbv,
            'acc_by_#nouns': gbn,
            'acc_by_#verbs': gbv2,
            'baseline': baseline([(len(c.n_spans), len(c.labels)) for c in cs]),
            'acc_by_rule': gbr,
            'acc_by_tree': gbt}


def verb_depth_acc(pss: list[list[int]], samples: list[CompactSample]) \
        -> tuple[dict[Any, tuple[int, int, float]], ...]:
    all_preds = [
        (' '.join([s for idx, s in enumerate(sample.sentence) if sample.v_spans[i][idx] == 1]),
         p == sample.labels[i], sample.depth, len(sample.n_spans), len(sample.v_spans))
        for sample, ps in zip(samples, pss) for i, p in enumerate(ps)]

    def gb(key: int):
        return [(k, [v[1] for v in vs]) for k, vs in groupby(sorted(all_preds, key=lambda x: x[key]), lambda x: x[key])]

    return ({k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gb(-3) if len(vs)},
            {k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gb(0) if len(vs)},
            {k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gb(-2) if len(vs)},
            {k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gb(-1) if len(vs)})


def rule_acc(pss: list[list[int]], samples: list[CompactSample]):
    def rule_to_v_ids(rule: Rule, tree: LabeledTree) -> list[tuple[str, int]]:
        return [(leaf[-1], v_id) for branch in labtree_to_branches(tree) if branch_to_rule(branch) == rule
                for leaf in branch[1] if (v_id := leaf[1]) is not None]

    all_preds = [(rule, [(eq(ps[i], sample.labels[i]), cat) for cat, i in rule_to_v_ids(rule, sample.labtree)])
                 for ps, sample in zip(pss, samples) for rule in abstree_to_rules(labeled_to_abstree(sample.labtree))]
    gbr = [(k, [cor for v in vs for cor, _ in v[1]]) for k, vs in groupby(sorted(all_preds, key=lambda x: x[0]),
                                                                          key=lambda x: x[0])]
    return {k: (c := sum(vs), ln := len(vs), c/ln) for k, vs in gbr if len(vs)}


def tree_acc(pss: list[list[int]], samples: list[CompactSample]):
    all_preds = [(sample.labtree, {i: (ps[i], sample.labels[i]) for i in range(len(sample.labels))})
                 for ps, sample in zip(pss, samples)]
    gbt = [(k, [v[1] for v in vs])
           for k, vs in groupby(sorted(all_preds, key=lambda x: str(labeled_to_abstree(x[0]))),
                                key=lambda x: str(labeled_to_abstree(x[0])))]
    return {k: {i: ([v[i][0] for v in vs], vs[0][1]) for i in vs[0]} for k, vs in gbt}


def consistency(trees: list[dict[int, tuple[list[int], int]]]):
    tree_consistency = [
        (grouped := sorted([(k, len(list(vs))) for k, vs in groupby(sorted(preds))],
                           key=lambda x: x[1], reverse=True),
         grouped[0][1] / sum([v for _, v in grouped]),
         grouped[0][0] == label)
        for tree in trees for node, (preds, ((label, _), _)) in tree.items()]
    return (
        (mu := sum([c for _, c, _ in tree_consistency]) / len(tree_consistency),
         sqrt(sum([(c-mu) ** 2 for _, c, _ in tree_consistency]) / len(tree_consistency))),
        len([t for _, c, t in tree_consistency if t]) / len(tree_consistency))



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
        if isinstance(xs[0], list):
            return sum(xs, [])
        raise TypeError(f'Cannot merge {type(xs[0])}')

    seeds = [results[s] for s in results if s != 'dataset']
    metrics = [k for k in seeds[0].keys()]
    return {**{metric: merge([seed[metric] for seed in seeds]) for metric in metrics},
            **{'dataset': results['dataset']}}
