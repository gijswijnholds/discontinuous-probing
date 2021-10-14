from transformers import AutoTokenizer, TOKENIZER_MAPPING
from .model_names import bertje_name
from .data_reader import CompactSample, read_grammar
from typing import NamedTuple
from torch.utils.data import Dataset


class ProcessedSample(NamedTuple):
    tokens:         list[int]
    verb_spans:     list[list[int]]
    noun_spans:     list[list[int]]
    compact:        CompactSample

    def check(self):
        assert len(self.verb_spans) == len(self.compact.labels)
        assert max(self.compact.labels) <= len(self.noun_spans) - 1
        assert set(map(len, self.verb_spans)) == set(map(len, self.noun_spans))
        assert set(sum(self.verb_spans, []) + sum(self.noun_spans, [])) == {0, 1}
        assert len(self.tokens) == len(self.verb_spans[0])
        assert self.tokens[0] == 1 and self.tokens[-1] == 2


def create_tokenizer(tokenizer_name=bertje_name) -> TOKENIZER_MAPPING:
    return AutoTokenizer.from_pretrained(tokenizer_name)


def expand_tags(word_tokens: list[list[str]], taglists: list[list[int]]) -> list[list[int]]:
    word_lens = list(map(len, word_tokens))
    return [[0] + sum([[tag] * word_lens[i] for i, tag in enumerate(taglist)], []) + [0] for taglist in taglists]


def capitalize_and_punctuate(vss: list[list[int]], nss: list[list[int]], ws: list[str]) \
        -> tuple[list[list[int]], list[list[int]], list[str]]:
    def capitalize(_iw: tuple[int, str]) -> str:
        _i, _w = _iw
        return _w if _i != 0 else _w[0].upper() + _w[1:]
    return [vs + [0] for vs in vss], [ns + [0] for ns in nss],  [capitalize(iw) for iw in enumerate(ws)] + ['.']


def tokenize_compact(
        tokenizer: TOKENIZER_MAPPING,
        compact_sample: CompactSample) -> ProcessedSample:
    vs, ns, ws = capitalize_and_punctuate(compact_sample.v_spans, compact_sample.n_spans, compact_sample.sentence)
    word_tokens = list(map(lambda w: tokenizer.tokenize(w), ws))
    noun_spans = expand_tags(word_tokens, ns)
    verb_spans = expand_tags(word_tokens, vs)
    tokens = [1] + tokenizer.convert_tokens_to_ids(sum(word_tokens, [])) + [2]
    return ProcessedSample(tokens, verb_spans, noun_spans, compact_sample)


def tokenize_compacts(
        tokenizer: TOKENIZER_MAPPING,
        data: list[CompactSample]) -> list[ProcessedSample]:
    def tokenize(sample: CompactSample):
        return tokenize_compact(tokenizer, sample)
    return [tokenize(sample) for sample in data]


class SpanDataset(Dataset):
    def __init__(self, data: list[ProcessedSample]):
        self.data = data
        for sample in self.data:
            sample.check()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> ProcessedSample:
        return self.data[i]


def prepare_datasets(fn: str) -> list[SpanDataset]:
    print("Preparing datasets...")
    datasets = read_grammar(fn)
    print("Getting tokenizer...")
    tokenizer = create_tokenizer()
    print("Tokenizing data...")
    return [SpanDataset(tokenize_compacts(tokenizer, dataset)) for dataset in datasets]
