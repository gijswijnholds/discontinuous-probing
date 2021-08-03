from typing import List, Tuple
import pickle
from pathlib import Path

Realized = list[tuple[list[int], list[int], str]]
example_fn = './data/grammars/example_control.p'


def open_grammar(fn: Path):
    with open(fn, 'rb') as inf:
        data = pickle.load(inf)
    return data['trees'], data['realizations'], data['matchings']


def realization_to_sequences(realization: Realized, matching: dict[int, int]):

    pass