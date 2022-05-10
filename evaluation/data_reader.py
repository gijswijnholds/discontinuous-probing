from typing import Optional as Maybe
from typing import Union, NamedTuple, Any
import json
import pickle

from random import shuffle, seed

Matchings = dict[int, list[int]]
Realized = list[tuple[list[int], list[int], str]]


LabeledTree = Union[tuple[Maybe[int], Maybe[int], str], tuple['LabeledTree', ...]]
AbsTree = Union[str, tuple['AbsTree', ...]]
Rule = tuple[str, tuple[str, ...]]


def labeled_to_abstree(ltree: LabeledTree) -> AbsTree:
    if len(ltree) == 3:
        return ltree[-1]
    root, children = ltree
    return labeled_to_abstree(root), tuple(map(labeled_to_abstree, children))


def get_aroot(_a: AbsTree) -> str:
    return _a if isinstance(_a, str) else _a[0]


def abstree_to_rules(atree: AbsTree) -> set[Rule]:
    if isinstance(atree, str):
        return set()
    root, children = atree
    return {(root, tuple(get_aroot(c) for c in children))}.union(rule for c in children for rule in abstree_to_rules(c))


def get_lroot(_ltree: LabeledTree) -> tuple[Maybe[int], Maybe[int], str]:
    return _ltree if len(_ltree) == 3 else _ltree[0]


def labtree_to_branches(ltree: LabeledTree) -> list[tuple[str, tuple[tuple[Maybe[int], Maybe[int], str], ...]]]:

    if len(ltree) == 3:
        return []
    root, children = ltree
    return [(root[-1], tuple(get_lroot(c) for c in children))] + sum([labtree_to_branches(c) for c in children], [])


def branch_to_rule(branch: tuple[str, tuple[tuple[Maybe[int], Maybe[int], str], ...]]) -> Rule:
    return branch[0], tuple(map(lambda c: c[-1], branch[1]))


class CompactSample(NamedTuple):
    depth:      Maybe[int]
    labtree:    Maybe[LabeledTree]
    sentence:   list[str]
    n_spans:    list[list[int]]
    v_spans:    list[list[int]]
    labels:     list[list[bool]]


def fix_matchings(matching: dict[str, list[Any]]) -> Matchings:
    return {int(k): list(map(int, vs)) for k, vs in matching.items()}


def process_grammar(
        data: dict[str, dict[str, tuple[dict[str, list[Any]], list[str]]]]) \
        -> list[tuple[int, LabeledTree, Matchings, Realized]]:
    return [(int(depth), eval(abstree), fix_matchings(data[depth][abstree][0]), eval(surface))
            for depth in data for abstree in data[depth] for surface in data[depth][abstree][1]]


def expand_spans(idss: list[list[int]], idx: int):
    return [1 if idx in ids else 0 for ids in idss]


def make_sample(depth: int, abstree: LabeledTree, matchings: Matchings, realization: Realized) -> CompactSample:
    def get_matchings(_verb: int) -> list[bool]:
        indexed_nouns = matchings[_verb] if _verb in matchings else []
        return [True if n in indexed_nouns else False for n in range(len(n_ids))]
    nss, vss, ws = zip(*realization)
    n_ids = set(sum(nss, []))
    v_ids = set(sum(vss, []))
    noun_spans = list(map(lambda ni: expand_spans(nss, ni), n_ids))
    verb_spans = list(map(lambda vi: expand_spans(vss, vi), v_ids))
    labels_out = list(*zip([get_matchings(verb) for verb in range(len(v_ids))]))
    return CompactSample(depth, abstree, ws, noun_spans, verb_spans, labels_out)


def makes_samples(subsets: list[list[tuple[int, LabeledTree, Matchings, Realized]]]) -> list[list[CompactSample]]:
    return [[make_sample(*s) for s in subset] for subset in subsets]


def read_grammar(grammar_fn: str) -> list[list[CompactSample]]:
    with open(grammar_fn, 'r') as inf:
        data = json.load(inf)
    return makes_samples([process_grammar(data)])


def read_lassy(lassy_fn: str) -> list[list[CompactSample]]:
    def transpose(xs: list[list[bool]]) -> list[list[bool]]:
        return [list(x) for x in zip(*xs)]

    with open(lassy_fn, 'rb') as f:
        raw = pickle.load(f)
    compacts = [(CompactSample(None, None, s, n, v, transpose(l)), f) for s, n, v, l, f in raw if len(v) > 0 and len(n) > 0]
    seed(42)
    shuffle(compacts)
    train, dev = compacts[:int(0.8 * len(compacts))], compacts[int(0.8 * len(compacts)):]
    print(f'{len(train)} training samples, of which {len(list(filter(lambda cf: cf[1], train)))} flagged')
    print(f'{len(dev)} dev samples, of which {len(list(filter(lambda cf: cf[1], dev)))} flagged')
    return [[t[0] for t in train], [t[0] for t in dev]]


def read_file(file: str) -> list[list[CompactSample]]:
    return read_grammar(file) if file.endswith('json') else read_lassy(file)
