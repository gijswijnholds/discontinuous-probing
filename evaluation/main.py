import os

import torch

from .sparse import SparseVA
from .preprocessing import prepare_datasets
from .trainer import Trainer, make_pretrainer, make_tester, Maybe, SpanDataset
from .analysis import analysis
from .model_names import bertje_name

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

GLOBAL_SEEDS = [3, 7, 42]


def setup_trainer(
        data_path: str,
        bert_name: str,
        freeze: bool,
        device: str,
        seed: int = 42,
        model_path: Maybe[str] = None) -> Trainer:
    torch.manual_seed(seed)
    model = SparseVA(bert_name=bert_name, freeze=freeze, dim=768, selection_h=128)
    datasets = prepare_datasets(data_path)
    if len(datasets) == 2:
        train_ds, val_ds = datasets
        return make_pretrainer(
            name=f'{data_path.split("/")[-1]}_{bert_name.split("/")[-1]}_{seed}',
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            batch_size_train=32,
            batch_size_val=128,
            optim_constructor=AdamW,
            lr=1e-04,
            loss_fn=CrossEntropyLoss(),
            device=device)
    # assert len(datasets) == 1
    test_ds = datasets[-1]
    # test_ds = sum(datasets, [])
    assert model_path is not None
    model.load(model_path)
    model.to(device)
    return make_tester(
        name=f'{bert_name.split("/")[-1]}_{seed}',
        model=model,
        test_dataset=test_ds,
        batch_size_test=128)


#
# def run_trainer(data_file: str, bert_name: str, freeze: bool, device: str = 'cuda', num_repeats: int = 1):
#     results = dict()
#     for i in range(num_repeats):
#         trainer = setup_trainer(data_file, bert_name, freeze, device=device, seed=GLOBAL_SEEDS[i])
#         results[i] = trainer.train_loop(80, val_every=5)
#     return results


def pretrain_probes(data_file: str, bert_name: str, device: str = 'cuda', num_repeats: int = 1):
    for i in range(num_repeats):
        trainer = setup_trainer(data_file, bert_name, True, device=device, seed=GLOBAL_SEEDS[i])
        _ = trainer.train_loop(80, val_every=5, save_at_best=True)


def test_probe(data_file: str, bert_name: str, weight_path: str, device: str = 'cuda') -> dict:
    _, seed, epoch = weight_path.split('/')[-1].split('_')
    print(f'Testing with seed {seed} @ epoch {epoch}')
    trainer = setup_trainer(data_file, bert_name, True, device=device, model_path=weight_path)
    return analysis(trainer.test_loader.dataset, trainer.predict_epoch())
