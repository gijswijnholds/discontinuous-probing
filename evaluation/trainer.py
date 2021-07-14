from tqdm import tqdm
import pdb
import torch
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch import LongTensor, Tensor, tensor, long, no_grad, stack, load, cat
from typing import List, Tuple, Callable, Any

Sample = Tuple[List[str], List[int], List[int], int]
Samples = List[Sample]


def sequence_collator(word_pad: int) -> Callable[[Samples], Tuple[LongTensor, LongTensor]]:
    def collate_fn(samples: Samples) -> tuple[Any, Any, Any, Any, Tensor]:
        input_ids, attention_masks, subj_spans, obj_spans, ys = list(zip(*samples))
        return (_pad_sequence([tensor(i, dtype=long) for i in input_ids], batch_first=True, padding_value=word_pad),
                _pad_sequence([tensor(m, dtype=long) for m in attention_masks], batch_first=True, padding_value=0),
                _pad_sequence([tensor(ss, dtype=long) for ss in subj_spans], batch_first=True, padding_value=0),
                _pad_sequence([tensor(os, dtype=long) for os in obj_spans], batch_first=True, padding_value=0),
                stack([tensor(y, dtype=long) for y in ys], dim=0))

    return collate_fn


def compute_accuracy(predictions, trues) -> float:
    # return torch.sum(trues == torch.round(predictions)) / float(len(predictions)) -> for a model that spits out a single value
    return torch.sum(trues == torch.argmax(predictions, axis=1)) / float(len(predictions))

class BertTrainer():
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
                 loss_fn: torch.nn.Module):
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                       shuffle=True, collate_fn=sequence_collator(word_pad))
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size_val,
                                     shuffle=True, collate_fn=sequence_collator(word_pad))
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size_test,
                                      shuffle=True, collate_fn=sequence_collator(word_pad))
        self.model = model
        self.optimizer = optim_constructor(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn

    def train_batch(self, input_ids: LongTensor, input_masks: LongTensor, ys: LongTensor):
        self.model.train()
        predictions = self.model.forward(input_ids, attention_mask=input_masks).logits
        batch_loss = self.loss_fn(predictions, ys)
        accuracy = compute_accuracy(predictions, ys)
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return batch_loss.item(), accuracy

    def train_epoch(self, device: str, epoch_i: int):
        epoch_loss, epoch_accuracy = 0., 0.
        batch_counter = 0
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for input_ids, input_masks, subj_tags, obj_tags, ys in tepoch:
                tepoch.set_description(f"Epoch {epoch_i}")

                input_ids.to(device)
                input_masks.to(device)
                ys.to(device)
                loss, accuracy = self.train_batch(input_ids, input_masks, ys)

                tepoch.set_postfix(loss=loss, accuracy=accuracy.item())

                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(self.train_loader), epoch_accuracy / len(self.train_loader)

    @no_grad()
    def eval_batch(self, input_ids: LongTensor, input_masks: LongTensor, ys: LongTensor):
        with torch.no_grad():
            self.model.eval()

            predictions = self.model.forward(input_ids, attention_mask=input_masks).logits
            batch_loss = self.loss_fn(predictions, ys)
            accuracy = compute_accuracy(predictions, ys)

        return batch_loss.item(), accuracy

    def eval_epoch(self, eval_set: str, device: str, epoch_i: int):
        epoch_loss, epoch_accuracy = 0., 0.
        loader = {'train': self.train_loader, 'val': self.val_loader, 'test': self.test_loader}[eval_set]
        batch_counter = 0
        with tqdm(loader, unit="batch") as tepoch:
            for input_ids, input_masks, subj_tags, obj_tags, ys in tepoch:
                tepoch.set_description(f"Epoch {epoch_i}")
                batch_counter += 1
                input_ids.to(device)
                input_masks.to(device)
                ys.to(device)
                loss, accuracy = self.eval_batch(input_ids, input_masks, ys)
                tepoch.set_postfix(loss=loss, accuracy=accuracy.item())
                # print(f"Batch {batch_counter}/{len(self.train_loader)}, loss: {loss:.5f}, acc: {accuracy:.5f}", end='\r')
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(loader), epoch_accuracy / len(loader)

class Trainer():
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
                 loss_fn: torch.nn.Module):
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                       shuffle=True, collate_fn=sequence_collator(word_pad))
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size_val,
                                     shuffle=True, collate_fn=sequence_collator(word_pad))
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size_test,
                                      shuffle=True, collate_fn=sequence_collator(word_pad))
        self.model = model
        self.optimizer = optim_constructor(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn

    def train_batch(self, input_ids: LongTensor, input_masks: LongTensor,
                    subj_tags: LongTensor, obj_tags: LongTensor, ys: LongTensor):
        self.model.train()
        predictions = self.model.forward(input_ids, input_masks, subj_tags, obj_tags)
        batch_loss = self.loss_fn(predictions, ys)
        accuracy = compute_accuracy(predictions, ys)

        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return batch_loss.item(), accuracy

    def train_epoch(self, device: str, epoch_i: int):
        epoch_loss, epoch_accuracy = 0., 0.
        batch_counter = 0
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for input_ids, input_masks, subj_tags, obj_tags, ys in tepoch:
                tepoch.set_description(f"Epoch {epoch_i}")

                input_ids.to(device)
                input_masks.to(device)
                subj_tags.to(device)
                obj_tags.to(device)
                ys.to(device)
                loss, accuracy = self.train_batch(input_ids, input_masks, subj_tags, obj_tags, ys)

                tepoch.set_postfix(loss=loss, accuracy=accuracy.item())

                # print(f"Batch {batch_counter}/{len(self.train_loader)}, loss: {loss:.5f}, acc: {accuracy:.5f}", end='\r')
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(self.train_loader), epoch_accuracy / len(self.train_loader)

    @no_grad()
    def eval_batch(self, input_ids: LongTensor, input_masks: LongTensor,
                   subj_tags: LongTensor, obj_tags: LongTensor, ys: LongTensor):
        self.model.eval()

        predictions = self.model.forward(input_ids, input_masks, subj_tags, obj_tags)
        batch_loss = self.loss_fn(predictions, ys)
        accuracy = compute_accuracy(predictions, ys)

        return batch_loss.item(), accuracy

    def eval_epoch(self, eval_set: str, device: str, epoch_i: int):
        epoch_loss, epoch_accuracy = 0., 0.
        loader = {'train': self.train_loader, 'val': self.val_loader, 'test': self.test_loader}[eval_set]
        batch_counter = 0
        with tqdm(loader, unit="batch") as tepoch:
            for input_ids, input_masks, subj_tags, obj_tags, ys in tepoch:
                tepoch.set_description(f"Epoch {epoch_i}")
                batch_counter += 1
                input_ids.to(device)
                input_masks.to(device)
                subj_tags.to(device)
                obj_tags.to(device)
                ys.to(device)
                loss, accuracy = self.eval_batch(input_ids, input_masks, subj_tags, obj_tags, ys)
                tepoch.set_postfix(loss=loss, accuracy=accuracy.item())
                # print(f"Batch {batch_counter}/{len(self.train_loader)}, loss: {loss:.5f}, acc: {accuracy:.5f}", end='\r')
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(loader), epoch_accuracy / len(loader)

    @no_grad()
    def predict_batch(self, input_ids: LongTensor, input_masks: LongTensor,
                      subj_tags: LongTensor, obj_tags: LongTensor, ys: LongTensor):
        self.model.eval()

        predictions = self.model.forward(input_ids, input_masks, subj_tags, obj_tags)
        batch_loss = self.loss_fn(predictions, ys)
        accuracy = compute_accuracy(predictions, ys)

        return batch_loss.item(), accuracy, input_ids, predictions

    def predict_epoch(self, eval_set: str, device: str, epoch_i: int):
        epoch_loss, epoch_accuracy = 0., 0.
        epoch_input_ids, epoch_predictions, epoch_ys = [], [], []
        loader = {'train': self.train_loader, 'val': self.val_loader, 'test': self.test_loader}[eval_set]
        batch_counter = 0
        with tqdm(loader, unit="batch") as tepoch:
            for input_ids, input_masks, subj_tags, obj_tags, ys in tepoch:
                tepoch.set_description(f"Epoch {epoch_i}")
                batch_counter += 1
                input_ids.to(device)
                input_masks.to(device)
                subj_tags.to(device)
                obj_tags.to(device)
                ys.to(device)
                loss, accuracy, input_ids, predictions = self.predict_batch(input_ids, input_masks, subj_tags, obj_tags, ys)
                tepoch.set_postfix(loss=loss, accuracy=accuracy.item())
                # print(f"Batch {batch_counter}/{len(self.train_loader)}, loss: {loss:.5f}, acc: {accuracy:.5f}", end='\r')
                epoch_loss += loss
                epoch_input_ids += input_ids
                epoch_accuracy += accuracy
                epoch_predictions += predictions
                epoch_ys += ys
        return epoch_loss / len(loader), epoch_accuracy / len(loader), epoch_input_ids, epoch_predictions, epoch_ys

    def main_loop(self, num_epochs: int, device: str):
        results = {i+1: {} for i in range(num_epochs)}
        for e in range(num_epochs):
            print(f"Epoch {e+1}...")
            train_loss, train_acc = self.train_epoch(device=device, epoch_i=e+1)
            print(f"Train loss {train_loss:.5f}, Train accuracy: {train_acc:.5f}")
            val_loss, val_acc = self.eval_epoch(eval_set='val', device=device, epoch_i=e+1)
            print(f"Val loss {val_loss:.5f}, Val accuracy: {val_acc:.5f}")
            test_loss, test_acc = self.eval_epoch(eval_set='test', device=device, epoch_i=e+1)
            print(f"Test loss {test_loss:.5f}, Test accuracy: {test_acc:.5f}")
            results[e+1] = {'train_loss': train_loss, 'train_acc': train_acc.item(),
                          'val_loss': val_loss, 'val_acc': val_acc.item(),
                          'test_loss': test_loss, 'test_acc': test_acc.item()}
        return results