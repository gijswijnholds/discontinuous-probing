import torch

from .sparse import SparseVA
from .preprocessing import prepare_datasets
from .trainer import Trainer

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

import os
from .model_names import bertje_name

GLOBAL_SEEDS = [3, 7, 42]


def setup_trainer(grammar_file: str, bert_name: str, freeze: bool, device: str, seed: int = 42) -> Trainer:
    torch.manual_seed(seed)
    model = SparseVA(bert_name=bert_name, freeze=freeze, dim=768, selection_h=128)
    train_ds, val_ds, test_ds = prepare_datasets(grammar_file)
    return Trainer(model=model,
                   train_dataset=train_ds,
                   val_dataset=val_ds,
                   test_dataset=test_ds,
                   batch_size_train=32, batch_size_val=128, batch_size_test=128,
                   word_pad=3,
                   optim_constructor=AdamW,
                   lr=1e-04,
                   loss_fn=CrossEntropyLoss(),
                   device=device)


def run_trainer(grammar_file: str, bert_name: str, freeze: bool, device: str = 'cuda', num_repeats: int = 1):
    results = dict()
    for i in range(num_repeats):
        trainer = setup_trainer(grammar_file, bert_name, freeze, device=device, seed=GLOBAL_SEEDS[i])
        results[i] = trainer.main_loop(80, val_every=5)
    return results


def do_everything(grammar_dir: str, bert_name: str = bertje_name, device: str = 'cuda'):
    results = dict()
    for file in os.listdir(grammar_dir):
        experiment = file.split('.')[0]
        print(experiment)
        results[experiment] = run_trainer(os.path.join(grammar_dir, file), bert_name=bert_name,
                                          freeze=True, device=device, num_repeats=3)
    return results


def __main__(gdir):
    import pickle
    with open('./results.p', 'wb') as f:
        pickle.dump(do_everything(gdir), f)
