import os

from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import LongTensor, Tensor, no_grad
from typing import Callable
from typing import Optional as Maybe
from .preprocessing import ProcessedSample, SpanDataset
from .sparse import sparse_matches, dense_matches, SparseVA


def sequence_collator(word_pad: int) -> \
    Callable[[list[ProcessedSample]],
             tuple[Tensor, Tensor, list[list[list[int]]], list[list[list[int]]], Tensor]]:

    def collate_fn(samples: list[ProcessedSample]) -> \
            tuple[Tensor, Tensor, list[list[list[int]]], list[list[list[int]]], Tensor]:
        input_ids = pad_sequence([torch.tensor(sample.tokens) for sample in samples],
                                 padding_value=word_pad, batch_first=True)
        input_mask = input_ids != word_pad
        verb_spanss = [sample.verb_spans for sample in samples]
        noun_spanss = [sample.noun_spans for sample in samples]
        labelss = [sample.compact.labels for sample in samples]
        return input_ids, input_mask, verb_spanss, noun_spanss, sparse_matches(noun_spanss, labelss)
    return collate_fn


def compute_accuracy(predictions: Tensor, trues: Tensor) -> float:
    return (torch.sum(trues == torch.argmax(predictions, dim=1)) / float(len(predictions))).item()


class Trainer:
    def __init__(self,
                 name: str,
                 model: SparseVA,
                 train_dataset: Maybe[SpanDataset] = None,
                 val_dataset: Maybe[SpanDataset] = None,
                 test_dataset: Maybe[SpanDataset] = None,
                 batch_size_train: Maybe[int] = None,
                 batch_size_val: Maybe[int] = None,
                 batch_size_test: Maybe[int] = None,
                 word_pad: int = 3,
                 optim_constructor: Maybe[type] = None,
                 lr: Maybe[float] = None,
                 loss_fn: Maybe[torch.nn.Module] = None,
                 device: str = 'cuda'):
        self.name = name
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                       collate_fn=sequence_collator(word_pad)) if train_dataset else None
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False,
                                     collate_fn=sequence_collator(word_pad)) if val_dataset else None
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                                      collate_fn=sequence_collator(word_pad)) if test_dataset else None
        self.model = model.to(device)
        self.optimizer = optim_constructor(self.model.parameters(), lr=lr) if optim_constructor else None
        self.loss_fn = loss_fn if loss_fn else None

    def train_batch(self, input_ids: LongTensor, input_masks: LongTensor, verb_spans: list[list[list[int]]],
                    noun_spans: list[list[list[int]]], ys: LongTensor) -> tuple[float, float]:
        self.model.train()

        predictions, _ = self.model.forward(input_ids, input_masks, verb_spans, noun_spans)
        batch_loss = self.loss_fn(predictions, ys)
        accuracy = compute_accuracy(predictions, ys)
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return batch_loss.item(), accuracy

    def train_epoch(self):
        epoch_loss, epoch_accuracy = 0., 0.
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for input_ids, input_masks, verb_spans, n_spans, ys in tepoch:
                loss, accuracy = self.train_batch(input_ids.to(self.device), input_masks.to(self.device),
                                                  verb_spans, n_spans, ys.to(self.device))
                tepoch.set_postfix(loss=loss, accuracy=accuracy)
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(self.train_loader), epoch_accuracy / len(self.train_loader)

    @no_grad()
    def eval_batch(self, input_ids: LongTensor, input_masks: LongTensor, verb_spans: list[list[list[int]]],
                   noun_spans: list[list[list[int]]], ys: LongTensor) -> tuple[float, float]:
        self.model.eval()

        predictions, _ = self.model.forward(input_ids, input_masks, verb_spans, noun_spans)
        batch_loss = self.loss_fn(predictions, ys)
        accuracy = compute_accuracy(predictions, ys)

        return batch_loss.item(), accuracy

    def eval_epoch(self, eval_set: str):
        epoch_loss, epoch_accuracy = 0., 0.
        loader = self.val_loader if eval_set == 'val' else self.test_loader
        batch_counter = 0
        with tqdm(loader, unit="batch") as tepoch:
            for input_ids, input_masks, verb_spans, n_spans, ys in tepoch:
                batch_counter += 1
                loss, accuracy = self.eval_batch(input_ids.to(self.device), input_masks.to(self.device),
                                                 verb_spans, n_spans, ys.to(self.device))
                tepoch.set_postfix(loss=loss, accuracy=accuracy)
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(loader), epoch_accuracy / len(loader)

    @no_grad()
    def predict_batch(self, input_ids: LongTensor, input_masks: LongTensor, verb_spans: list[list[list[int]]],
                      noun_spans: list[list[list[int]]]) -> list[list[int]]:
        self.model.eval()
        predictions, vn_mask = self.model.forward(input_ids, input_masks, verb_spans, noun_spans)
        return dense_matches(predictions, vn_mask)

    @no_grad()
    def predict_epoch(self) -> list[list[int]]:
        return [label
                for input_ids, input_masks, v_spans, n_spans, _ in self.test_loader
                for label in self.predict_batch(input_ids.to(self.device), input_masks.to(self.device),
                                                v_spans, n_spans)]

    def train_loop(self, num_epochs: int, val_every: int = 1, save_at_best: bool = False):
        results = dict()
        for e in range(num_epochs):
            print(f"Epoch {e}...")
            train_loss, train_acc = self.train_epoch()
            print(f"Train loss {train_loss:.5f}, Train accuracy: {train_acc:.5f}")
            if (e % val_every == 0 and e != 0) or e == num_epochs - 1:
                val_loss, val_acc = self.eval_epoch(eval_set='val')
                print(f"Val loss {val_loss:.5f}, Val accuracy: {val_acc:.5f}")
                if save_at_best and val_acc > max([v['val_acc'] for v in results.values()]):
                    for file in os.listdir('./'):
                        if file.startswith(f'{self.name}'):
                            os.remove(file)
                    self.model.save(f'{self.name}_{e}')
            else:
                val_loss, val_acc = None, -1
            results[e] = {'train_loss': train_loss, 'train_acc': train_acc,
                          'val_loss': val_loss, 'val_acc': val_acc}
        print(f"Best epoch was {max(results, key=lambda k: results[k]['val_acc'])}")
        return results


def make_pretrainer(
        name: str,
        model: SparseVA,
        train_dataset: SpanDataset,
        val_dataset: SpanDataset,
        batch_size_train: int,
        batch_size_val: int,
        optim_constructor: type,
        lr: float,
        loss_fn: torch.nn.Module,
        device: str = 'cuda') -> Trainer:
    return Trainer(name=name, model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                   batch_size_train=batch_size_train, batch_size_val=batch_size_val,
                   optim_constructor=optim_constructor, lr=lr, loss_fn=loss_fn, device=device)


def make_tester(
        name: str,
        model: SparseVA,
        test_dataset: SpanDataset,
        batch_size_test: int) -> Trainer:
    return Trainer(name=name, model=model, test_dataset=test_dataset, batch_size_test=batch_size_test)