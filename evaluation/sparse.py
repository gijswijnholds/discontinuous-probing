import torch
from torch import Tensor
from random import randint
from torch_geometric.nn import global_mean_pool as placeholder_aggregation

from typing import Sequence


def test():
    from torch.nn.utils.rnn import pad_sequence
    batch_size = 32
    max_seq_len = 50
    max_nouns = 5
    max_verbs = 3
    dim = 128
    sent_lens = [randint(3, max_seq_len) for _ in range(batch_size)]
    mask = pad_sequence([torch.ones(sent_len, dtype=torch.long) for sent_len in sent_lens], batch_first=True)
    bert_outputs = torch.rand((batch_size, mask.shape[1], dim))
    n_spans: list[list[list[int]]] = [[[randint(0, 1) for s in range(sent_len)]
                                       for _ in range(randint(1, max_nouns))] for sent_len in sent_lens]
    v_spans: list[list[list[int]]] = [[[randint(0, 1) for s in range(sent_len)]
                                       for _ in range(randint(1, max_nouns))] for sent_len in sent_lens]
    find_matches(bert_outputs, mask, n_spans, v_spans)


def find_matches(bert_outputs: Tensor,
                 mask: Tensor,
                 n_spanss: list[list[list[int]]],
                 v_spans: list[list[list[int]]])\
        -> Tensor:

    offsets = mask.sum(dim=-1)                      # (batch,)                  -- how many words per sentence
    offsets = offsets.cumsum(dim=-1).roll(1, 0)     # (batch,)
    offsets[0] = 0                                  # (batch,)                  -- start of each sentence
    sparse_batch = bert_outputs[mask != 0]          # (num_tokens, dim))        -- token reprs for the entire batch
    n_batch, n_id, n_source, num_n = sparse_spans(n_spanss, offsets)
    v_batch, v_id, v_source, num_v = sparse_spans(v_spans, offsets)
    n_reprs = placeholder_aggregation(
        sparse_batch[n_source], n_id)               # (num_nouns, dim)          -- aggregated noun representations
    v_reprs = placeholder_aggregation(
        sparse_batch[v_source], v_id)               # (num_verbs, dim)          -- aggregated verb representations
    v_mask, n_mask = torch.cartesian_prod(v_batch, n_batch).chunk(2, dim=-1)
    vn_mask = v_mask.eq(n_mask).view(num_v, num_n)  # (num_verbs, num_nouns)    -- verb to noun alignment
    x_atn: Tensor = ...                             # (num_verbs, num_nouns)    -- verb to noun attention
    x_atn[vn_mask.eq(0)] = -1e10
    return x_atn.softmax(dim=-1)


def sparse_spans(spanss: list[list[list[int]]], offsets: Sequence[int]) -> tuple[Tensor, Tensor, Tensor, int]:
    split = [(batch, span) for batch, spans in enumerate(spanss) for span in spans]
    batch_idx, token_idx, spans = list(zip(*[
        (b, torch.ones(sum(span), dtype=torch.long) * idx, torch.tensor(span).nonzero().squeeze(-1) + offsets[b])
        for idx, (b, span) in enumerate(split)]))
    return torch.tensor(batch_idx), torch.cat(token_idx), torch.cat(spans), len(split)


def batch_matches(n_spans: list[list[list[int]]], v_spans: list[list[list[int]]], labels: list[list[int]]) -> Tensor:
    """

    :param n_spans:
    :param v_spans:
    :param labels:
    :return:                LongTensor of size (num_verbs,) with values from [0, num_nouns]
    """
    ...

