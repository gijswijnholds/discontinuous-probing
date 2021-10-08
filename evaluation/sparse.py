import torch
from torch import Tensor
from torch_geometric.nn import global_mean_pool as placeholder_aggregation
from transformers import AutoModel
from typing import Sequence, Callable


def find_matches(bert_outputs: Tensor,
                 mask: Tensor,
                 v_spanss: list[list[list[int]]],
                 n_spanss: list[list[list[int]]],
                 x_atn_fn: Callable[[Tensor, Tensor, Tensor], Tensor]) \
        -> Tensor:

    offsets = mask.sum(dim=-1)                      # (batch,)                  -- how many words per sentence
    offsets = offsets.cumsum(dim=-1).roll(1, 0)     # (batch,)
    offsets[0] = 0                                  # (batch,)                  -- start of each sentence
    sparse_batch = bert_outputs[mask != 0]          # (num_tokens, dim)         -- token reprs for the entire batch
    n_batch, n_id, n_source, num_n = sparse_spans(n_spanss, offsets)
    v_batch, v_id, v_source, num_v = sparse_spans(v_spanss, offsets)
    n_reprs = placeholder_aggregation(
        sparse_batch[n_source], n_id)               # (num_nouns, dim)          -- aggregated noun representations
    v_reprs = placeholder_aggregation(
        sparse_batch[v_source], v_id)               # (num_verbs, dim)          -- aggregated verb representations
    v_mask, n_mask = torch.cartesian_prod(v_batch, n_batch).chunk(2, dim=-1)
    vn_mask = v_mask.eq(n_mask).view(num_v, num_n)  # (num_verbs, num_nouns)    -- verb to noun alignment
    return x_atn_fn(v_reprs, n_reprs, vn_mask)      # (num_verbs, num_nouns)    -- verb to noun attention


def sparse_spans(spanss: list[list[list[int]]], offsets: Sequence[int]) -> tuple[Tensor, Tensor, Tensor, int]:
    split = [(batch, span) for batch, spans in enumerate(spanss) for span in spans]
    batch_idx, token_idx, spans = list(zip(*[
        (b, torch.ones(sum(span), dtype=torch.long) * idx, torch.tensor(span).nonzero().squeeze(-1) + offsets[b])
        for idx, (b, span) in enumerate(split)]))
    return torch.tensor(batch_idx), torch.cat(token_idx), torch.cat(spans), len(split)


def sparse_matches(n_spanss: list[list[list[int]]], labels: list[list[int]]) -> Tensor:
    offsets = torch.tensor([len(n_spans) for n_spans in n_spanss], dtype=torch.long).cumsum(dim=-1).roll(1, 0)
    offsets[0] = 0
    return torch.tensor([noun_id + offsets[b] for b, sentence in enumerate(labels) for noun_id in sentence])


class SparseAtn(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super(SparseAtn, self).__init__()
        self.selection_q = torch.nn.Linear(dim, hidden, bias=False)
        self.selection_k = torch.nn.Linear(dim, hidden, bias=False)

    def forward(self, qs: Tensor, ks: Tensor, mask: Tensor):
        qs = self.selection_q(qs)                   # Q x H
        ks = self.selection_k(ks)                   # K x H
        x_atn = qs @ ks.t()                         # Q x K
        x_atn[mask.eq(0)] = -1e-10
        return x_atn


class SparseVA(torch.nn.Module):
    def __init__(self, dim: int, span_h: int, num_heads: int, selection_h: int, model_name: str, freeze: bool = True):
        super(SparseVA, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        if self.freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
        assert span_h % num_heads == 0
        self.x_atn = SparseAtn(dim, selection_h)

    def forward(self, input_ids, input_masks, v_spanss: list[list[list[int]]], n_spanss: list[list[list[int]]]):
        embeddings = self.bert_model(input_ids, attention_mask=input_masks)[0]                  # B x S x D
        return find_matches(embeddings, input_masks, v_spanss, n_spanss, self.x_atn)            # (num_verbs, num_nouns)
