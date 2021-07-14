from evaluation.tokenizer import create_tokenizer, tokenize_string_with_spans
from evaluation.task import WinoVerbNL, WinoVerbNLSynth
import torch
from torch.utils.data import Dataset
import pickle


def separate_spans(tagss):
    return zip(*map(lambda l: ([1 if x == 1 else 0 for x in l], [1 if x == 2 else 0 for x in l]), tagss))


def get_tokenized_sents_labels(tokenizer, data, sentence_cutoff=60):
    """There is one sentence that is super long and pushes the max token length to 232, so we can remove a single
    sentence and push down the token length to 190."""
    inputss = [tokenize_string_with_spans(tokenizer, d['sentence'].split(), d['tags'])
               for d in data if len(d['sentence'].split()) < sentence_cutoff and 1 in d['tags']]
    input_idss, input_tagss, input_maskss = zip(*inputss)
    input_subj_tagss, input_obj_tagss = separate_spans(input_tagss)
    labels = [d['label'] for d in data
              if len(d['sentence'].split()) < sentence_cutoff]
    return {'input_ids': list(input_idss),
            'input_subj_tags': list(input_subj_tagss),
            'input_obj_tags': list(input_obj_tagss),
            'attention_mask': list(input_maskss),
            'label': list(labels)}


def prepare_dataset(name=None, sentence_cutoff=60, token_max=140, task=WinoVerbNL):
    print("Preparing dataset...")
    dataset = task(name=name).data
    if name:
        out_fn = 'dataset_' + name + '_' + str(sentence_cutoff) + '_' + str(token_max) + '.p'
    else:
        out_fn = 'dataset_' + str(sentence_cutoff) + '_' + str(token_max) + '.p'
    print("Getting tokenizer...")
    tokenizer = create_tokenizer()
    print("Tokenizing data...")
    tokenized_train = get_tokenized_sents_labels(tokenizer, dataset['train'], sentence_cutoff=sentence_cutoff)
    tokenized_val = get_tokenized_sents_labels(tokenizer, dataset['val'], sentence_cutoff=sentence_cutoff)
    tokenized_test = get_tokenized_sents_labels(tokenizer, dataset['test'], sentence_cutoff=sentence_cutoff)
    with open(out_fn, 'wb') as out_file:
        pickle.dump((tokenized_train, tokenized_val, tokenized_test), out_file)
    return (tokenized_train, tokenized_val, tokenized_test)


def load_prepped_dataset(name=None, sentence_cutoff=60, token_max=140):
    if name:
        fn = 'dataset_' + name + '_' + str(sentence_cutoff) + '_' + str(token_max) + '.p'
    else:
        fn = 'dataset' + '_' + str(sentence_cutoff) + '_' + str(token_max) + '.p'
    with open(fn, 'rb') as datafile:
        data = pickle.load(datafile)
    return data


def preprocess_dataset(name=None, already_prepped=True, sentence_cutoff=60, max_len=140):
    if already_prepped:
        train_tokenized, val_tokenized, test_tokenized = load_prepped_dataset(name, sentence_cutoff=sentence_cutoff,
                                                                              token_max=max_len)
    else:
        train_tokenized, val_tokenized, test_tokenized = prepare_dataset(name, sentence_cutoff=sentence_cutoff,
                                                                         token_max=max_len)
    return train_tokenized, val_tokenized, test_tokenized


class SpanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, i):
        return (self.data['input_ids'][i], self.data['attention_mask'][i],
                self.data['input_subj_tags'][i], self.data['input_obj_tags'][i], self.data['label'][i])


class SpanDatasetFullSpans(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, i):
        return (self.data['input_ids'][i], self.data['attention_mask'][i],
                [1]*len(self.data['input_subj_tags'][i]), [1]*len(self.data['input_obj_tags'][i]), self.data['label'][i])


def get_tokenized_sents_labels_verb(tokenizer, data, sentence_cutoff=60):
    """There is one sentence that is super long and pushes the max token length to 232, so we can remove a single
    sentence and push down the token length to 190."""
    inputss = [tokenize_string_with_spans(tokenizer, d['sentence'].split(), d['tags'])
               for d in data if len(d['sentence'].split()) < sentence_cutoff and 1 in d['tags']]
    input_idss, input_tagss, input_maskss = zip(*inputss)
    labels = [d['label'] for d in data
              if len(d['sentence'].split()) < sentence_cutoff]
    return {'input_ids': list(input_idss),
            'input_verb_tags': list(input_tagss),
            'attention_mask': list(input_maskss),
            'label': list(labels)}



def prepare_dataset_verb(sentence_cutoff=60, token_max=140):
    print("Preparing dataset...")
    dataset = WinoVerbNL(name='verb').data
    out_fn = 'dataset_verb_' + str(sentence_cutoff) + '_' + str(token_max) + '.p'

    print("Getting tokenizer...")
    tokenizer = create_tokenizer()
    print("Tokenizing data...")
    tokenized_train = get_tokenized_sents_labels_verb(tokenizer, dataset['train'], sentence_cutoff=sentence_cutoff)
    tokenized_val = get_tokenized_sents_labels_verb(tokenizer, dataset['val'], sentence_cutoff=sentence_cutoff)
    tokenized_test = get_tokenized_sents_labels_verb(tokenizer, dataset['test'], sentence_cutoff=sentence_cutoff)
    with open(out_fn, 'wb') as out_file:
        pickle.dump((tokenized_train, tokenized_val, tokenized_test), out_file)
    return tokenized_train, tokenized_val, tokenized_test


def preprocess_dataset_verb(already_prepped=True, sentence_cutoff=60, max_len=140):
    if already_prepped:
        train_tokenized, val_tokenized, test_tokenized = load_prepped_dataset('verb', sentence_cutoff=sentence_cutoff,
                                                                              token_max=max_len)
    else:
        train_tokenized, val_tokenized, test_tokenized = prepare_dataset_verb(sentence_cutoff=sentence_cutoff,
                                                                         token_max=max_len)
    return train_tokenized, val_tokenized, test_tokenized



def preprocess_dataset_synth(name=None, already_prepped=True, sentence_cutoff=60, max_len=140):
    if already_prepped:
        train_tokenized, val_tokenized, test_tokenized = load_prepped_dataset(name, sentence_cutoff=sentence_cutoff,
                                                                              token_max=max_len)
    else:
        train_tokenized, val_tokenized, test_tokenized = prepare_dataset(name, sentence_cutoff=sentence_cutoff,
                                                                         token_max=max_len, task=WinoVerbNLSynth)
    return train_tokenized, val_tokenized, test_tokenized
