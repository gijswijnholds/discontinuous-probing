from typing import List, Tuple
from config import VERB_IDX
from evaluation.task import WinoVerbNLSynth
from transformers import AutoTokenizer
from evaluation.model_names import bertje_name
from torch.utils.data import Dataset


def create_tokenizer(tokenizer_name=bertje_name):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def get_expanded_tags(word_tokens: List[List[str]], tags: List[int]):
    """Given a list of lists of tokenized words and a list of tags corresponding to each word, we expand the tags
    according to the length of each word tokenization."""
    return sum(map(lambda wts_tg: len(wts_tg[0])*[wts_tg[1]], zip(word_tokens, tags)), [])

def tokenize_string_with_spans(tokenizer, words: List[str], noun_spans: List[List[int]], verb_span: List[int]):
    # todo
    # Noun spans and verb span should have contain 0's and 1's instead of 0's and their idx.
    word_tokens = map(lambda w: tokenizer.tokenize(w), words)
    expanded_noun_spans = map(lambda span: get_expanded_tags(word_tokens, span), noun_spans)
    expanded_verb_span = get_expanded_tags(word_tokens, verb_span)
    word_tokens = sum(word_tokens, [])
    tokens = tokenizer.convert_tokens_to_ids(word_tokens)
    expanded_noun_spans = map(lambda span: [0] + span + [0])
    expanded_verb_span = [0] + expanded_verb_span + [0]
    input_ids = [1] + tokens + [2]
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, expanded_verb_span, expanded_noun_spans

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


def prepare_dataset(name=None):
    # todo
    # Update to load from grammars files rather than a class.
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