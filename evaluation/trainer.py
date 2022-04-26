import os

from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import LongTensor, Tensor, no_grad
from typing import Callable, Any
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


def compute_stats(predictions: Tensor, trues: Tensor) -> dict[str, float]:
    rounded = predictions.round()
    true_positives = torch.sum(rounded * trues).item()
    predicted_positives = torch.sum(rounded).item()
    all_relevants = torch.sum(trues).item()
    precision = true_positives / predicted_positives
    recall = true_positives / all_relevants
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = rounded.eq(trues).sum().item() / len(rounded)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}


class Trainer:
    def __init__(self,
                 name: str,
                 model: SparseVA,
                 word_pad: int,
                 train_dataset: Maybe[SpanDataset] = None,
                 val_dataset: Maybe[SpanDataset] = None,
                 test_dataset: Maybe[SpanDataset] = None,
                 batch_size_train: Maybe[int] = None,
                 batch_size_val: Maybe[int] = None,
                 batch_size_test: Maybe[int] = None,
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

    def train_batch(
            self,
            batch: tuple[LongTensor, LongTensor, list[list[list[int]]], list[[list[int]]], LongTensor]) \
            -> tuple[float, dict[str, float], Tensor, Tensor]:
        self.model.train()
        input_ids, input_masks, verb_spans, noun_spans, ys = batch
        predictions, output_mask = self.model.forward(
            input_ids.to(self.device), input_masks.to(self.device), verb_spans, noun_spans)
        batch_loss = self.loss_fn(predictions := predictions[output_mask], ys := ys[output_mask].to(self.device))
        stats = compute_stats(predictions, ys)
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return batch_loss.item(), accuracy

    def train_epoch(self) -> tuple[float, dict[str, float]]:
        epoch_loss, epoch_preds, epoch_trues = 0., torch.tensor([]), torch.tensor([])
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                loss, stats, preds, trues = self.train_batch(batch)
                tepoch.set_postfix(loss=loss, stats=stats)
                epoch_loss += loss
                epoch_preds = torch.cat((epoch_preds, preds), 1)
                epoch_trues = torch.cat((epoch_trues, trues), 1)
        return epoch_loss / len(self.train_loader), compute_stats(epoch_preds, epoch_trues)

    @no_grad()
    def eval_batch(
            self,
            batch: tuple[LongTensor, LongTensor, list[list[list[int]]], list[[list[int]]], LongTensor]) \
            -> tuple[float, Tensor, Tensor]:
        self.model.eval()
        input_ids, input_masks, verb_spans, noun_spans, ys = batch
        predictions, output_mask = self.model.forward(
            input_ids.to(self.device), input_masks.to(self.device), verb_spans, noun_spans)
        batch_loss = self.loss_fn(predictions, ys := ys.to(self.device))
        accuracy = compute_accuracy(predictions, ys)

        return batch_loss.item(), accuracy

    def eval_epoch(self, eval_set: str):
        epoch_loss, epoch_preds, epoch_trues = 0., torch.tensor([]), torch.tensor([])
        loader = self.val_loader if eval_set == 'val' else self.test_loader
        batch_counter = 0
        with tqdm(loader, unit="batch") as tepoch:
            for batch in tepoch:
                batch_counter += 1
                loss, accuracy = self.eval_batch(batch)
                tepoch.set_postfix(loss=loss, accuracy=accuracy)
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(loader), epoch_accuracy / len(loader)

    @no_grad()
    def predict_batch(
            self,
            batch: tuple[LongTensor, LongTensor, list[list[list[int]]], list[[list[int]]], Any]) \
            -> list[list[int]]:
        self.model.eval()
        input_ids, input_masks, verb_spans, noun_spans, _ = batch
        predictions, vn_mask = self.model.forward(
            input_ids.to(self.device), input_masks.to(self.device), verb_spans, noun_spans)
        return dense_matches(predictions, vn_mask)

    @no_grad()
    def predict_epoch(self) -> list[list[int]]:
        return [label for batch in self.test_loader for label in self.predict_batch(batch)]

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
                val_loss, val_stats = None, {'loss': -1, 'f1': -1, 'precision': -1, 'recall': -1}
            results[e] = {'train_loss': train_loss,
                          'train_precision': train_stats['precision'],
                          'train_recall': train_stats['recall'],
                          'train_f1': train_stats['f1'],
                          'val_loss': val_loss,
                          'val_precision': val_stats['precision'],
                          'val_recall': val_stats['recall'],
                          'val_f1': val_stats['f1']}
        print(f"Best epoch was {max(results, key=lambda k: results[k]['val_f1'])}")
        return results


def make_pretrainer(
        name: str,
        model: SparseVA,
        word_pad_id: int,
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
                   optim_constructor=optim_constructor, lr=lr, loss_fn=loss_fn, device=device, word_pad=word_pad_id)


def make_tester(
        name: str,
        model: SparseVA,
        word_pad_id: int,
        test_dataset: SpanDataset,
        batch_size_test: int,
        device: str) -> Trainer:
    return Trainer(name=name, model=model, test_dataset=test_dataset,
                   batch_size_test=batch_size_test, device=device, word_pad=word_pad_id)
