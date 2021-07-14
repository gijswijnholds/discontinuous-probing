import pickle
import pdb
import torch
from evaluation.model import SpanClassModel, SpanClassModelAttention, SpanClassModelAttentionVerb, SpanClassModelAttentionN
from evaluation.preprocessor import preprocess_dataset, SpanDataset, SpanDatasetFullSpans, preprocess_dataset_verb, preprocess_dataset_synth
from evaluation.trainer import Trainer
from evaluation.analyse import analyse_test
from config import train_configs


def write_results_config(config: dict, results: dict):
    name, optim, lr, loss_fn = config['name'].__str__(), config['optim'].__name__, config['lr'], config['loss_fn'].__name__
    bert_model, freeze, run = config['bert_model'].replace('/','--'), config['freeze'].__str__(), config['run']
    batch_size = config['batch_size']
    style = config['style']
    fn = f"./results/results_style={style}_name={name}_bertmodel={bert_model}_freeze={freeze}_optim={optim}_lr={lr}_bs={batch_size}_loss={loss_fn}_run={run}.p"
    new_dict = {}
    for e in results:
        new_dict[f'results_{e}'] = results[e]
    for c in config:
        new_dict[f'config_{c}'] = config[c]
    write_results(fn=fn, results=new_dict)


def write_results(fn: str, results: dict):
    with open(fn, 'wb') as out_file:
        pickle.dump(results, out_file)


def train_model(config):
    name, optim_constructor, lr, loss_fn = config['name'], config['optim'], config['lr'], config['loss_fn']
    freeze, epochs, batch_size, bert_model = config['freeze'], config['epochs'], config['batch_size'], config['bert_model']
    if config['style'] == 'Attention':
        my_model = SpanClassModelAttention(dim=768, model_name=bert_model, freeze=freeze)
        dataset = preprocess_dataset(name=name, already_prepped=False, sentence_cutoff=60, max_len=140)
        train_dataset, val_dataset, test_dataset = SpanDataset(dataset[0]), SpanDataset(dataset[1]), SpanDataset(
            dataset[2])
    elif config['style'] == 'AttentionVerb':
        my_model = SpanClassModelAttentionVerb(dim=768, model_name=bert_model, freeze=freeze)
        dataset = preprocess_dataset_verb(already_prepped=False, sentence_cutoff=60, max_len=140)
        train_dataset, val_dataset, test_dataset = SpanDataset(dataset[0]), SpanDataset(dataset[1]), SpanDataset(
            dataset[2])
    elif config['style'] == 'AttentionFullSpan':
        my_model = SpanClassModelAttention(dim=768, model_name=bert_model, freeze=freeze)
        dataset = preprocess_dataset(name=name, already_prepped=False, sentence_cutoff=60, max_len=140)
        train_dataset, val_dataset, test_dataset = SpanDatasetFullSpans(dataset[0]), SpanDatasetFullSpans(dataset[1]), SpanDatasetFullSpans(dataset[2])
    elif config['style'] == 'SynthAttention':
        print("Using a synthetic dataset!")
        # my_model = SpanClassModelAttentionN(dim=768, spans=2, model_name=bert_model, freeze=freeze)
        my_model = SpanClassModelAttention(dim=768, model_name=bert_model, freeze=freeze)
        dataset = preprocess_dataset_synth(name=name, already_prepped=False, sentence_cutoff=60, max_len=140)
        train_dataset, val_dataset, test_dataset = SpanDataset(dataset[0]), SpanDataset(dataset[1]), SpanDataset(
            dataset[2])
    else:
        my_model = SpanClassModel(dim=768, model_name=bert_model, freeze=freeze)
        dataset = preprocess_dataset(name=name, already_prepped=False, sentence_cutoff=60, max_len=140)
        train_dataset, val_dataset, test_dataset = SpanDataset(dataset[0]), SpanDataset(dataset[1]), SpanDataset(
            dataset[2])


    my_trainer = Trainer(model=my_model,
                         train_dataset=train_dataset,
                         val_dataset=val_dataset,
                         test_dataset=test_dataset,
                         batch_size_train=batch_size, batch_size_val=batch_size, batch_size_test=batch_size, word_pad=3,
                         optim_constructor=optim_constructor, lr=lr, loss_fn=loss_fn())
                         # optim_constructor=torch.optim.AdamW, lr=0.00003, loss_fn=torch.nn.CrossEntropyLoss())

    # results = my_trainer.main_loop(num_epochs=epochs, device='cpu')
    # pdb.set_trace()
    # return results, my_trainer
    return my_trainer
    # total_metrics, orig_metrics, swap_metrics = analyse_test(my_trainer)
    # return results, total_metrics, orig_metrics, swap_metrics

    # write_results_config(config=config, results=results)
    # return results


if __name__ == '__main__':
    for config in train_configs:
        train_model(config=config)