from transformers import AutoTokenizer
from evaluation.model_names import bertje_name
from typing import List

def get_seq(seq_ten):
    return seq_ten[seq_ten > 3]

def untokenize(tokenizer, seq):
    return ' '.join(tokenizer.convert_ids_to_tokens(seq))

def create_tokenizer(tokenizer_name=bertje_name):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_string(tokenizer, input: str, max_len: int):
    pad_id = tokenizer.pad_token_id
    input_ids = tokenizer.encode(input)
    input_len, padding_len = len(input_ids), max_len - len(input_ids)
    input_ids = input_ids + ([pad_id] * padding_len)
    attention_mask = ([1] * input_len) + ([0] * padding_len)
    return input_ids, attention_mask


def tokenize_string_with_spans(tokenizer, words: List[str], tags: List[int]):
    word_toks = map(lambda w: tokenizer.tokenize(w), words)
    word_toks_tags = map(lambda wts_tg: (wts_tg[0], [wts_tg[1]]*len(wts_tg[0])), zip(word_toks, tags))
    word_toks, tags = map(lambda i: sum(i, []), zip(*word_toks_tags))
    toks = tokenizer.convert_tokens_to_ids(word_toks)
    input_ids, input_tags = [1] + toks + [2], [0] + tags + [0]
    attention_mask = [1] * len(input_ids)
    return input_ids, input_tags, attention_mask