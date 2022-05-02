import pdb

import torch
from torch import Tensor
from transformers import AutoModel
from typing import Callable
from torch_geometric.nn import GlobalAttention
from torch.nn import Dropout
from torch.nn.utils.rnn import pad_sequence
from itertools import groupby


def find_matches(bert_outputs: Tensor,
                 mask: Tensor,
                 v_spanss: list[list[list[int]]],
                 n_spanss: list[list[list[int]]],
                 verb_aggr_fn: Callable[[Tensor, Tensor], Tensor],
                 noun_aggr_fn: Callable[[Tensor, Tensor], Tensor],
                 x_atn_fn: Callable[[Tensor, Tensor, Tensor], Tensor]) \
        -> tuple[Tensor, Tensor]:

    offsets = mask.sum(dim=-1)                      # (batch,)                  -- how many words per sentence
    offsets = offsets.cumsum(dim=-1).roll(1, 0)     # (batch,)
    offsets[0] = 0                                  # (batch,)                  -- start of each sentence
    sparse_batch = bert_outputs[mask != 0]          # (num_tokens, dim)         -- token reprs for the entire batch
    n_batch, n_id, n_source, num_n = sparse_spans(n_spanss, offsets)
    v_batch, v_id, v_source, num_v = sparse_spans(v_spanss, offsets)
    n_reprs = noun_aggr_fn(
        sparse_batch[n_source], n_id)               # (num_nouns, dim)          -- aggregated noun representations
    v_reprs = verb_aggr_fn(
        sparse_batch[v_source], v_id)               # (num_verbs, dim)          -- aggregated verb representations
    v_mask, n_mask = torch.cartesian_prod(v_batch, n_batch).chunk(2, dim=-1)
    vn_mask = v_mask.eq(n_mask).view(num_v, num_n)  # (num_verbs, num_nouns)    -- verb to noun alignment
    return x_atn_fn(v_reprs, n_reprs, vn_mask), vn_mask


def sparse_spans(spanss: list[list[list[int]]], offsets: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
    def ones(x: int) -> Tensor:
        return torch.ones(x, dtype=torch.long, device=offsets.device)
    split = [(batch, span) for batch, spans in enumerate(spanss) for span in spans]
    batch_idx, token_idx, spans = list(zip(*[
        (b, ones(sum(span)) * idx, torch.tensor(span, device=offsets.device).nonzero().squeeze(-1) + offsets[b])
        for idx, (b, span) in enumerate(split)]))
    return torch.tensor(batch_idx, device=offsets.device), torch.cat(token_idx), torch.cat(spans), len(split)


def sparse_matches(labelss: list[list[list[bool]]]) -> Tensor:
    num_nouns_per_sent = [len(labels[0]) for labels in labelss]

    def suffix(i: int) -> list[bool]: return [False] * sum(num_nouns_per_sent[:i])
    rows = [suffix(i) + verb for i, sentence in enumerate(labelss) for verb in sentence]
    return pad_sequence([torch.tensor(row, dtype=torch.float) for row in rows], batch_first=True)


def dense_matches(predictions: Tensor, vn_mask: Tensor) -> list[list[list[bool]]]:
    """
    :param predictions: Verbs x Nouns tensor of scores
    :param vn_mask: Verbs x Nouns tensor of boolean values
    :return: num_sents x num_verbs_per_sent x num_nouns_per_sent
    """
    rounded = predictions.round().bool()
    num_verbs_per_sent = [len(list(g)) for _, g in groupby(vn_mask.tolist())]
    all_predictions = [rounded[i][verb_mask.ne(0)].tolist() for i, verb_mask in enumerate(vn_mask)]
    offset = 0
    return [all_predictions[offset:(offset := offset + num_v)] for num_v in num_verbs_per_sent]


class SparseAtn(torch.nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super(SparseAtn, self).__init__()
        self.selection_q = torch.nn.Linear(dim, hidden, bias=False)
        self.selection_k = torch.nn.Linear(dim, hidden, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, qs: Tensor, ks: Tensor, mask: Tensor):
        qs = self.dropout(self.selection_q(qs))     # Q x H
        ks = self.dropout(self.selection_k(ks))     # K x H
        x_atn = qs @ ks.t()                         # Q x K
        x_atn[mask.eq(0)] = -1e10
        return x_atn


class SparseVA(torch.nn.Module):
    def __init__(self, dim: int, selection_h: int, bert_name: str, freeze: bool, dropout: float = 0.1):
        super(SparseVA, self).__init__()
        self.bert_model = AutoModel.from_pretrained(bert_name)
        self.freeze = freeze
        if self.freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
        self.v_aggr = GlobalAttention(gate_nn=torch.nn.Linear(dim, 1))
        self.n_aggr = GlobalAttention(gate_nn=torch.nn.Linear(dim, 1))
        self.x_atn = SparseAtn(dim, selection_h, dropout)
        self.dropout = Dropout(dropout)

    def save(self, fn: str):
        if not self.freeze:
            torch.save({k: v for k, v in self.state_dict().items()}, fn)
        else:
            torch.save({k: v for k, v in self.state_dict().items() if 'bert_model' not in k}, fn)

    def load(self, fn):
        sdict = torch.load(fn, map_location='cpu')
        self.load_state_dict(sdict, strict=False)

    def forward(
            self,
            input_ids: Tensor,
            input_masks: Tensor,
            v_spanss: list[list[list[int]]], n_spanss: list[list[list[int]]]) -> tuple[Tensor, Tensor]:
        embeddings = self.dropout(self.bert_model(input_ids, attention_mask=input_masks)[0])        # B x S x D
        return find_matches(
            bert_outputs=embeddings,
            mask=input_masks,
            v_spanss=v_spanss,
            n_spanss=n_spanss,
            verb_aggr_fn=self.v_aggr,
            noun_aggr_fn=self.n_aggr,
            x_atn_fn=self.x_atn)
