import os
from pathlib import Path

_DESCRIPTION = """\
The WinoVerbNL is a Winograd-like task in which a model is tasked with identifying the understood subject in
a verbal complement, depending on the choice of control verb in the main clause.
"""


class WinoVerbNL():
    def __init__(self, folder='./data', name=None):
        data_dir = Path(folder)

        def add_name(s: str):
            if name:
                return s.split('.')[0] + '_' + name + '.txt'
            else:
                return s
        self.data = {
                     'train': self.load_examples(path=data_dir / add_name('all_task_data_train.txt')),
                     'val': self.load_examples(path=data_dir / add_name('all_task_data_val.txt')),
                     'test': self.load_examples(path=data_dir / add_name('all_task_data_test.txt')),
                    }

    def load_examples(self, path):
        """Generator of examples for each split."""
        data = []
        with open(path) as in_file:
            for idx, line in enumerate(in_file):
                sent, tags, label = line.split('\t')
                data.append({
                    'sentence': sent,
                    'tags': list(map(int, tags.split())),
                    'label': int(label)-1,
                })
        return data


class WinoVerbNLSynth():
    def __init__(self, folder='./data', name=None):
        data_dir = Path(folder)

        def add_name(s: str):
            if name:
                return s.split('.')[0] + '_' + name + '.txt'
            else:
                return s
        self.data = {
                     'train': self.load_examples(path=data_dir / add_name('synth_task_data_train.txt')),
                     'val': self.load_examples(path=data_dir / add_name('synth_task_data_val.txt')),
                     'test': self.load_examples(path=data_dir / add_name('synth_task_data_test.txt')),
                    }

    def load_examples(self, path):
        """Generator of examples for each split."""
        data = []
        with open(path) as in_file:
            for idx, line in enumerate(in_file):
                sent, tags, label = line.split('\t')
                data.append({
                    'sentence': sent,
                    'tags': list(map(int, tags.split())),
                    'label': int(label)-1,
                })
        return data