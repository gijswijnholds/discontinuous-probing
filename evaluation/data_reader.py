from typing import Optional as Maybe
from typing import Union, NamedTuple
import json
import pickle

Matching = dict[int, int]
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
    labels:     list[int]


def fix_matching(matching: dict[str, int]) -> Matching:
    return {int(k): matching[k] for k in matching}


def process_grammar(
        data: dict[str, dict[str, tuple[dict[str, int], list[str]]]]) \
        -> list[tuple[int, LabeledTree, Matching, Realized]]:
    return [(int(depth), eval(abstree), fix_matching(data[depth][abstree][0]), eval(surface))
            for depth in data for abstree in data[depth] for surface in data[depth][abstree][1]]


def expand_spans(idss: list[list[int]], idx: int):
    return [1 if idx in ids else 0 for ids in idss]


def make_sample(depth: int, abstree: LabeledTree, matching: Matching, realization: Realized) -> CompactSample:
    nss, vss, ws = zip(*realization)
    n_ids = set(sum(nss, []))
    v_ids = set(sum(vss, []))
    noun_spans = list(map(lambda ni: expand_spans(nss, ni), n_ids))
    verb_spans = list(map(lambda vi: expand_spans(vss, vi), v_ids))
    labels_out = list(map(lambda k: matching[k], matching))
    return CompactSample(depth, abstree, ws, noun_spans, verb_spans, labels_out)


def makes_samples(subsets: list[list[tuple[int, LabeledTree, Matching, Realized]]]) -> list[list[CompactSample]]:
    return [[make_sample(*s) for s in subset] for subset in subsets]


def read_grammar(grammar_fn: str) -> list[list[CompactSample]]:
    with open(grammar_fn, 'r') as inf:
        data = json.load(inf)
    return makes_samples([process_grammar(data)])


def read_lassy(lassy_fn: str) -> list[list[CompactSample]]:
    with open(lassy_fn, 'rb') as f:
        subsets = pickle.load(f)
    return [[CompactSample(None, None, s, n, v, l) for s, n, v, l in subset]
            for subset in subsets]


def read_file(file: str) -> list[list[CompactSample]]:
    return read_grammar(file) if file.endswith('json') else read_lassy(file)
