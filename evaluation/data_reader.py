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


def open_grammar(fn: str) -> list[tuple[int, AbsTree, Matching, Realized]]:
    with open(fn, 'r') as inf:
        data = json.load(inf)
    return [(int(depth), abstree, fix_matching(data[depth][abstree][0]), eval(surface))
            for depth in data for abstree in data[depth] for surface in data[depth][abstree][1]]


def get_span(constant: str, idx: Maybe[int]) -> list[int]:
    return len(constant.split()) * [idx]


def get_full_span(wss: list[str], idss: list[list[int]], idx: int):
    """
    ["de socialist", "eet", "een appel"] -> [0, 0, None, 1, 1]
                                        -> [[1, 1, 0, 0, 0], [0, 0, 0, 1, 1]]
    """
    return sum([get_span(ws, idx if idx in ids else None) for (ids, ws) in zip(idss, wss)], [])


def correct_indices(depth: int, abstree: AbsTree, sentence: list[str], noun_spans: list[list[int]],
                    verb_spans: list[list[int]], labels: list[int]) -> CompactSample:
    """Replace None indices by 0 and offset all other indices"""
    noun_spans_out = list(map(lambda span: [0 if i is None else 1 for i in span], noun_spans))
    verb_spans_out = list(map(lambda span: [0 if v is None else 1 for v in span], verb_spans))
    labels_out = list(map(lambda label: label+1, labels))
    return CompactSample(depth, abstree, sentence, noun_spans_out, verb_spans_out, labels_out)


def make_sample(depth: int, abstree: AbsTree, matching: Matching, realization: Realized) -> CompactSample:
    """Given a realization (a list of constituents with their noun/verb indications, a matching from verbs to nouns,
    and the special index for the verb, we generate multiple data samples (one for each verb), for the model to train
    on. The format is: (sentence, noun spans, [(verb_span, label), ...])"""
    nss, vss, wss = zip(*realization)
    n_ids = set(sum(nss, []))
    noun_spans = list(map(lambda ni: get_full_span(wss, nss, ni), n_ids))
    verb_spans = list(map(lambda k: get_full_span(wss, vss, k), matching))
    labels = list(map(lambda k: matching[k], matching))
    return correct_indices(depth, abstree, wss, noun_spans, verb_spans, labels)


def read_grammar(grammar_fn: str) -> list[CompactSample]:
    samples = open_grammar(grammar_fn)
    return list(map(lambda s: make_sample(*s), samples))
