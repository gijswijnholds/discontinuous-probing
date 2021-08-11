import pickle
import pdb
import torch
from evaluation.attention_model import VerbArgumentAttention
from evaluation.attention_preprocessor import prepare_dataset, SpanDataset
from evaluation.attention_trainer import Trainer
from evaluation.attention_grammarreader import train_example_fn, dev_example_fn, test_example_fn
from config import train_configs


def setup_trainer(config):
    name, optim_constructor, lr, loss_fn = config['name'], config['optim'], config['lr'], config['loss_fn']
    freeze, epochs, batch_size, bert_model = config['freeze'], config['epochs'], config['batch_size'], config['bert_model']

    my_model = VerbArgumentAttention(dim=768, span_h=200, num_heads=4, selection_h=100,
                                     model_name=bert_model, freeze=freeze)
    dataset = prepare_dataset(train_grammar_fn=train_example_fn, dev_grammar_fn=dev_example_fn, test_grammar_fn=test_example_fn)
    train_dataset, val_dataset, test_dataset = SpanDataset(dataset[0]), SpanDataset(dataset[1]), SpanDataset(
        dataset[2])

    trainer = Trainer(model=my_model,
                      train_dataset=train_dataset,
                      val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      batch_size_train=batch_size, batch_size_val=batch_size, batch_size_test=batch_size, word_pad=3,
                      optim_constructor=optim_constructor, lr=lr, loss_fn=loss_fn())
    return trainer


def setup_trainer_vice_versa(config):
    name, optim_constructor, lr, loss_fn = config['name'], config['optim'], config['lr'], config['loss_fn']
    freeze, epochs, batch_size, bert_model = config['freeze'], config['epochs'], config['batch_size'], config['bert_model']

    my_model = VerbArgumentAttention(dim=768, span_h=200, num_heads=4, selection_h=100,
                                     model_name=bert_model, freeze=freeze)
    dataset = prepare_dataset(grammar_fn=example_fn)
    train_dataset, val_dataset, test_dataset = SpanDataset(dataset[0]), SpanDataset(dataset[1]), SpanDataset(
        dataset[2])

    trainer = Trainer(model=my_model,
                      train_dataset=val_dataset,
                      val_dataset=train_dataset,
                      test_dataset=test_dataset,
                      batch_size_train=batch_size, batch_size_val=batch_size, batch_size_test=batch_size, word_pad=3,
                      optim_constructor=optim_constructor, lr=lr, loss_fn=loss_fn())
    return trainer


def train_model(config):
    trainer = setup_trainer(config)
    print(config['name'])
    return trainer.main_loop(num_epochs=config['epochs'], device='cpu')


def train_model_vice_versa(config):
    trainer = setup_trainer_vice_versa(config)
    print(config['name'])
    return trainer.main_loop(num_epochs=config['epochs'], device='cpu')


if __name__ == '__main__':
    results_vice_versa = {config['name']: train_model_vice_versa(config) for config in train_configs}
    results = {config['name']: train_model(config) for config in train_configs}
