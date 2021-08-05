from typing import List, Tuple
import random
from config import VERB_IDX
from evaluation.task import WinoVerbNLSynth
from transformers import AutoTokenizer
from evaluation.model_names import bertje_name
from torch.utils.data import Dataset
from evaluation.attention_grammarreader import grammar_to_dataset_format

def create_tokenizer(tokenizer_name=bertje_name):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def get_expanded_tags(word_tokens: List[List[str]], tags: List[int]):
    """Given a list of lists of tokenized words and a list of tags corresponding to each word, we expand the tags
    according to the length of each word tokenization."""
    return sum(map(lambda wts_tg: len(wts_tg[0])*[wts_tg[1]], zip(word_tokens, tags)), [])

def tokenize_string_with_spans(tokenizer, words: List[str], noun_spans: List[List[int]], verb_span: List[int]):
    word_tokens = list(map(lambda w: tokenizer.tokenize(w), words))
    expanded_noun_spans = list(map(lambda span: get_expanded_tags(word_tokens, list(span)), noun_spans))
    expanded_noun_spans = list(map(lambda span: list(map(lambda i: 1 if i > 0 else 0, span)), expanded_noun_spans))
    expanded_verb_span = get_expanded_tags(word_tokens, verb_span)
    expanded_verb_span = list(map(lambda i: 1 if i > 0 else 0, expanded_verb_span))
    word_tokens = sum(word_tokens, [])
    tokens = tokenizer.convert_tokens_to_ids(word_tokens)
    expanded_noun_spans = list(map(lambda span: [0] + list(span) + [0], expanded_noun_spans))
    expanded_verb_span = [0] + expanded_verb_span + [0]
    input_ids = [1] + tokens + [2]
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, expanded_verb_span, expanded_noun_spans


def get_tokenized_sents_labels(tokenizer, data):
    """Tokenize inputs and separate their spans."""
    inputss = map(lambda d: tokenize_string_with_spans(tokenizer, d['sentence'].split(), d['noun_spans'], d['verb_span']), data)
    input_idss, input_maskss, verb_spanss, spansss = zip(*inputss)
    labels = [d['label'] for d in data]
    return {'input_ids': list(input_idss),
            'attention_mask': list(input_maskss),
            'verb_span': list(verb_spanss),
            'spans': list(spansss),
            'label': list(labels)}


def prepare_dataset(grammar_fn: str):
    print("Preparing dataset...")
    dataset = grammar_to_dataset_format(grammar_fn)
    num_train, num_val, num_test = (50000, 5000, 5000)
    random.shuffle(dataset, random.seed(238597))
    train_data = dataset[:num_train]
    val_data = dataset[num_train:num_train+num_val]
    test_data = dataset[num_train+num_val:num_train+num_val+num_test]
    print("Getting tokenizer...")
    tokenizer = create_tokenizer()
    print("Tokenizing data...")
    tokenized_train = get_tokenized_sents_labels(tokenizer, train_data)
    tokenized_val = get_tokenized_sents_labels(tokenizer, val_data)
    tokenized_test = get_tokenized_sents_labels(tokenizer, test_data)
    return (tokenized_train, tokenized_val, tokenized_test)


class SpanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, i):
        return (self.data['input_ids'][i], self.data['attention_mask'][i],
                self.data['verb_span'][i], self.data['spans'][i], self.data['label'][i])