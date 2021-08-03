from typing import List, Tuple
from config import VERB_IDX
from evaluation.task import WinoVerbNLSynth
from transformers import AutoTokenizer
from evaluation.model_names import bertje_name
from torch.utils.data import Dataset


def create_tokenizer(tokenizer_name=bertje_name):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_string_with_spans(tokenizer, words: List[str], tags: List[int]):
    word_toks = map(lambda w: tokenizer.tokenize(w), words)
    word_toks_tags = map(lambda wts_tg: (wts_tg[0], [wts_tg[1]]*len(wts_tg[0])), zip(word_toks, tags))
    word_toks, tags = map(lambda i: sum(i, []), zip(*word_toks_tags))
    toks = tokenizer.convert_tokens_to_ids(word_toks)
    input_ids, input_tags = [1] + toks + [2], [0] + tags + [0]
    attention_mask = [1] * len(input_ids)
    return input_ids, input_tags, attention_mask


def separate_spans(multispan: List[int], verb_idx: int) -> Tuple[List[int], List[List[int]]]:
    """Given a list of integers indicating multiple candidates, a verb, and rest of the sentence,
        split these into multiple spans (one for the verb phrase, and one for each of the candidates).
        Example: [0, 0, 1, 1, 0, 2, 3, 3, 100, 100] should return
                 ([0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]]"""
    span_idxs = list(set([i for i in multispan if i > 0 and not i == verb_idx]))
    verb_span = [1 if x == verb_idx else 0 for x in multispan]
    spans = [[1 if x == span_idx else 0 for x in multispan] for span_idx in span_idxs]
    return verb_span, spans


def tokenize_and_separate_sents(tokenizer, d):
    input_ids, input_tags, attention_mask = tokenize_string_with_spans(tokenizer, d['sentence'].split(), d['tags'])
    verb_span, spans = separate_spans(input_tags, VERB_IDX)
    return input_ids, attention_mask, verb_span, spans


def get_tokenized_sents_labels(tokenizer, data):
    """Tokenize inputs and separate their spans."""
    inputss = [tokenize_and_separate_sents(tokenizer, d) for d in data]
    input_idss, input_maskss, verb_spanss, spansss = zip(*inputss)
    labels = [d['label'] for d in data]
    return {'input_ids': list(input_idss),
            'attention_mask': list(input_maskss),
            'verb_span': list(verb_spanss),
            'spans': list(spansss),
            'label': list(labels)}


def prepare_dataset(name=None):
    print("Preparing dataset...")
    dataset = WinoVerbNLSynth(name=name).data
    print("Getting tokenizer...")
    tokenizer = create_tokenizer()
    print("Tokenizing data...")
    tokenized_train = get_tokenized_sents_labels(tokenizer, dataset['train'])
    tokenized_val = get_tokenized_sents_labels(tokenizer, dataset['val'])
    tokenized_test = get_tokenized_sents_labels(tokenizer, dataset['test'])
    return (tokenized_train, tokenized_val, tokenized_test)


class SpanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, i):
        return (self.data['input_ids'][i], self.data['attention_mask'][i],
                self.data['verb_span'][i], self.data['spans'][i], self.data['label'][i])

def preprocess_dataset():
    NotImplemented