from typing import Optional as Maybe
from typing import Union, NamedTuple
import json

Matching = dict[int, int]
Realized = list[tuple[list[int], list[int], str]]
example_fn = './data/grammars/example_control.p'
# train_example_fn = './data/grammars/example_cluster_train.txt'
# dev_example_fn = './data/grammars/example_cluster_dev.txt'
# test_example_fn = './data/grammars/example_cluster_test.txt'
train_example_fn = './data/grammars/example_control_train.txt'
dev_example_fn = './data/grammars/example_control_dev.txt'
test_example_fn = './data/grammars/example_control_test.txt'


AbsTree = Union[tuple[Maybe[int], Maybe[int], str], tuple['AbsTree', ...]]


class CompactSample(NamedTuple):
    depth:      int
    abstree:    AbsTree
    sentence:   list[str]
    n_spans:    list[list[int]]
    v_spans:    list[list[int]]
    labels:     list[int]


def fix_matching(matching: dict[str, int]) -> Matching:
    return {int(k): matching[k] for k in matching}


def process_grammar(
        data: dict[str, dict[str, dict[AbsTree, tuple[dict[str, int], list[str]]]]]) \
        -> list[list[tuple[int, AbsTree, Matching, Realized]]]:
    return [[(int(depth), abstree, fix_matching(data[subset][depth][abstree][0]), eval(surface))
             for depth in data[subset]
             for abstree in data[subset][depth]
             for surface in data[subset][depth][abstree][1]]
            for subset in data]


def expand_spans(idss: list[list[int]], idx: int):
    return [1 if idx in ids else 0 for ids in idss]


def capitalize_and_punctuate(
        nss: list[list[int]],
        vss: list[list[int]],
        ws: list[str]) -> tuple[list[list[int]], list[list[int]], list[str]]:
    def capitalize(_iw: tuple[int, str]) -> str:
        _i, _w = _iw
        return _w if _i != 0 else _w[0].upper() + _w[1:]
    return [ns + [0] for ns in nss], [vs + [0] for vs in vss], [capitalize(iw) for iw in enumerate(ws)] + ['.']


def make_sample(depth: int, abstree: AbsTree, matching: Matching, realization: Realized) -> CompactSample:
    """Given a realization (a list of constituents with their noun/verb indications, a matching from verbs to nouns,
    and the special index for the verb, we generate multiple data samples (one for each verb), for the model to train
    on. The format is: (sentence, noun spans, [(verb_span, label), ...])"""
    nss, vss, ws = zip(*realization)
    n_ids = set(sum(nss, []))
    v_ids = set(sum(vss, []))
    noun_spans = list(map(lambda ni: expand_spans(nss, ni), n_ids))
    verb_spans = list(map(lambda vi: expand_spans(vss, vi), v_ids))
    labels_out = list(map(lambda k: matching[k], matching))
    nss, vss, ws = capitalize_and_punctuate(nss, vss, ws)
    return CompactSample(depth, abstree, ws, noun_spans, verb_spans, labels_out)


def makes_samples(subsets: list[list[tuple[int, AbsTree, Matching, Realized]]]) -> list[list[CompactSample]]:
    return [[make_sample(*s) for s in subset] for subset in subsets]


def read_grammar(grammar_fn: str) -> list[list[CompactSample]]:
    with open(grammar_fn, 'r') as inf:
        data = json.load(inf)
    return makes_samples(process_grammar(data))

