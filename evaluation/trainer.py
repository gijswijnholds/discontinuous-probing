from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch import LongTensor, Tensor, no_grad
from typing import Callable
from .preprocessing import ProcessedSample
from .sparse import sparse_matches, dense_matches


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
                 model: torch.nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 test_dataset: Dataset,
                 batch_size_train: int,
                 batch_size_val: int,
                 batch_size_test: int,
                 word_pad: int,
                 optim_constructor: type,
                 lr: float,
                 loss_fn: torch.nn.Module,
                 device: str):
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                       shuffle=True, collate_fn=sequence_collator(word_pad))
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size_val,
                                     shuffle=False, collate_fn=sequence_collator(word_pad))
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size_test,
                                      shuffle=False, collate_fn=sequence_collator(word_pad))
        self.model = model.to(device)
        self.optimizer = optim_constructor(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn

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

    def train_epoch(self, epoch_i: int):
        epoch_loss, epoch_accuracy = 0., 0.
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for input_ids, input_masks, verb_spans, n_spans, ys in tepoch:
                tepoch.set_description(f"Epoch {epoch_i}")

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

    def eval_epoch(self, eval_set: str, epoch_i: int):
        epoch_loss, epoch_accuracy = 0., 0.
        loader = self.val_loader if eval_set == 'val' else self.test_loader
        batch_counter = 0
        with tqdm(loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch_i}")
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

    def main_loop(self, num_epochs: int, val_every: int = 1):
        results = {i+1: {} for i in range(num_epochs)}
        for e in range(num_epochs):
            print(f"Epoch {e+1}...")
            train_loss, train_acc = self.train_epoch(epoch_i=e+1)
            print(f"Train loss {train_loss:.5f}, Train accuracy: {train_acc:.5f}")
            if ((e+1) % val_every == 0 and e != 0) or e == (num_epochs - 1):
                val_loss, val_acc = self.eval_epoch(eval_set='val', epoch_i=e+1)
                print(f"Val loss {val_loss:.5f}, Val accuracy: {val_acc:.5f}")
                test_preds = self.predict_epoch()
            else:
                val_loss, val_acc, test_preds = None, -1, None
            results[e+1] = {'train_loss': train_loss, 'train_acc': train_acc,
                            'val_loss': val_loss, 'val_acc': val_acc,
                            'test_preds': test_preds}
        print(f"Best epoch was {max(results, key=lambda k: results[k]['val_acc'])}")
        results['test_data'] = self.test_loader.dataset
        return results
