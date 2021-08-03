from typing import List, Tuple
from typing import Optional as Maybe
import pickle
from pathlib import Path

Realized = list[tuple[list[int], list[int], str]]
example_fn = './data/grammars/example_control.p'
CompactSamples = tuple[List[str], List[List[int]], List[List[int]], int]

def open_grammar(fn: Path):
    with open(fn, 'rb') as inf:
        data = pickle.load(inf)
    return data['trees'], data['realizations'], data['matchings']


def get_span(constant: str, idx: Maybe[int]) -> List[int]:
    return len(constant.split()) * [idx]


def get_full_span(wss: List[str], idss: List[List[int]], idx: int):
    return sum([get_span(ws, idx if idx in ids else None) for (ids, ws) in zip(idss, wss)], [])


def correct_indices(samples: CompactSamples, special_idx: int) -> CompactSamples:
    """Replace None indices by 0 and offset all other indices"""
    sentence, noun_spans, verb_items = samples
    noun_spans_out = list(map(lambda span: [0 if i is None else i+1 for i in span], noun_spans))
    verb_items_out = list(map(lambda span_label: ([0 if v is None else special_idx for v in span_label[0]],
                                                  span_label[1]+1), verb_items))
    return sentence, noun_spans_out, verb_items_out


def realization_to_sequences(realization: Realized, matching: dict[int, int], special_idx: int)\
        -> CompactSamples:
    """Given a realization (a list of constituents with their noun/verb indications, a matching from verbs to nouns,
    and the special index for the verb, we generate multiple data samples (one for each verb), for the model to train
    on. The format is: (sentence, noun spans, [(verb_span, label), ...])"""
    nss, vss, wss = zip(*realization)
    sentence = ' '.join(wss)
    n_ids = set(sum(nss, []))
    v_ids = set(sum(vss, []))
    noun_spans = list(map(lambda ni: get_full_span(wss, nss, ni), n_ids))
    verb_items = list(map(lambda k: (get_full_span(wss, vss, k), matching[k]), matching))
    compact_samples = (sentence, noun_spans, verb_items)
    return correct_indices(compact_samples, special_idx)



def main():
    trees, reals, matchings = open_grammar(example_fn)
