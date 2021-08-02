import torch
from math import sqrt
from transformers import AutoModel
from opt_einsum import contract


class CandidateAttention(torch.nn.Module):
    """Computes dot product attention *weights* for a single query and a number of keys."""
    def __init__(self, dim: int, hidden: int):
        super(CandidateAttention, self).__init__()
        self.selection_q = torch.nn.Linear(dim, hidden, bias=False)
        self.selection_k = torch.nn.Linear(dim, hidden, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, verb, candidates, candidate_masks):
        """
            :param verb: [batch_size, dim]
            :param candidates: [batch_size, num_candidates, dim]
            :param candidate_masks [batch_size, num_candidates]
            :returns: the masked (softmaxed) attention weights that show what is the most likely candidate for the verb.
        """
        verb_q = self.selection_q(verb)                 # B x 1 x H
        candidates_k = self.selection_k(candidates)     # B x C x H
        attn_weights = torch.bmm(verb_q.unsqueeze(1), candidates_k.transpose(1, 2)).squeeze() # B x C
        attn_weights[candidate_masks.eq(0)] = -1e10      # B x C
        # candidate_scores = self.softmax(attn_weights) # B x C
        # return candidate_scores
        return attn_weights

class SpanAttention(torch.nn.Module):
    """Computes a span representation using a sort of attention (but it's just a projection..)."""
    def __init__(self, dim: int, hidden: int, num_heads: int):
        super(SpanAttention, self).__init__()
        self.hidden = hidden
        self.attn_dim = hidden // num_heads
        self.num_heads = num_heads
        self.weight_proj = torch.nn.Linear(dim, 1, bias=False)
        self.q_transformation = torch.nn.Linear(dim, hidden, bias=False)
        self.k_transformation = torch.nn.Linear(dim, hidden, bias=False)
        self.v_transformation = torch.nn.Linear(dim, hidden, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, span_embeddings, span_tags):
        """
            :param span_embeddings: [batch_size, max_seq_len, dim]
            :param span_tags: [batch_size, max_seq_len]
            :return: a weighted sum of a projection of the input span embeddings
        """
        b, s, _ = span_embeddings.shape
        attn_weights = self.weight_proj(span_embeddings) # B x S x 1
        attn_weights[span_tags.eq(0)] = -1e10            # B x S x 1
        attn_weights = self.softmax(attn_weights)        # B x S x 1
        qs = self.q_transformation(span_embeddings).view(-1, s, self.attn_dim, self.num_heads)
        ks = self.k_transformation(span_embeddings).view(-1, s, self.attn_dim, self.num_heads)
        vs = self.v_transformation(span_embeddings).view(-1, s, self.attn_dim, self.num_heads)

        # relevance_weights = torch.bmm(qs, ks.transpose(1, 2)) / sqrt(self.hidden)
        relevance_weights = contract('bqdh,bkdh->bqkh', qs, ks) / sqrt(self.attn_dim) # B x S x S x H
        relevance_weights[span_tags.eq(0)] = -1e10
        relevance_weights = relevance_weights.softmax(dim=-1)
        # span_value = torch.bmm(relevance_weights, vs) # B x S x H
        span_value = contract('bqkh,bkdh->bqdh', relevance_weights, vs).flatten(-2, -1) # B x S x H
        weighted_value = (attn_weights * span_value).sum(dim=1).squeeze(1) # B x H
        return weighted_value


class VerbArgumentAttention(torch.nn.Module):
    """Computes a distribution over candidates of some verb in a sentence (argument selection)."""
    def __init__(self, dim: int, span_h: int, num_heads: int, selection_h: int, model_name: str, freeze: bool = True):
        super(VerbArgumentAttention, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        if self.freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
        assert span_h % num_heads == 0
        self.verb_embedder = SpanAttention(dim=dim, hidden=span_h, num_heads=num_heads)
        self.span_embedder = SpanAttention(dim=dim, hidden=span_h, num_heads=num_heads)
        self.candidate_attn = CandidateAttention(span_h, selection_h)

    def forward(self, input_ids, input_masks, verb_tags, candidate_masks, candidate_tags):
        """
            :param input_ids: [batch_size, max_seq_len]
            :param input_masks: [batch_size, max_seq_len]
            :param verb_tags: [batch_size, max_seq_len]
            :param candidate_masks: [batch_size, num_candidates]
            :param candidate_tags: [num_candidates, [batch_size, max_seq_len] ]
            :return: a classification embedding: [batch_size, num_candidates]
        """
        embeddings = self.bert_model(input_ids, attention_mask=input_masks)[0]      # B x S x D
        verb_embedding = self.verb_embedder(embeddings, verb_tags)                  # B x D
        candidate_embeddings = torch.stack([self.span_embedder(embeddings, tags) for tags in candidate_tags], dim=1)    # B x C x D
        candidate_scores = self.candidate_attn(verb_embedding, candidate_embeddings, candidate_masks)
        return candidate_scores

    """   
              0.verb_tensor [batch_size, dim]                   # representation of vp for each sentence in batch
              1.a) for every candidate n -> [batch_size, dim]      # representation of candidate n for each sentence in batch
                b) -> [batch_size, n, dim] 
              2. a mask of shape [batch_size, n]                 # mask[b, i] says whether candidate i is in sentence b
            we get scores [batch_size,  n]                  # weighting for each candidate i per sentence b
            scores[mask == False] -> -infinity              # mask away invalid candidates
            ... softmax, same shape 
            ... goal [batch_size] with values 0..n          # 
        """

def test_data():
    """
                :param input_ids: [batch_size, max_seq_len]
                :param input_masks: [batch_size, max_seq_len]
                :param verb_tags: [batch_size, max_seq_len]
                :param candidate_masks: [batch_size, num_candidates]
                :param candidate_tags: [batch_size, num_candidates, max_seq_len]
                :return: a classification embedding: [batch_size, num_candidates]
    """
    batch_size, max_seq_len, num_candidates = 10, 20, 5
    dim, span_h, num_heads, selection_h = 768, 200, 4, 100
    input_ids = torch.randint(low=3, high=2400, size=(batch_size, max_seq_len))
    input_masks = torch.randint(low=0, high=2, size=(batch_size, max_seq_len))
    verb_tags = torch.randint(low=0, high=2, size=(batch_size, max_seq_len))
    candidate_masks = torch.randint(low=0, high=2, size=(batch_size, num_candidates))
    candidate_tags = torch.randint(low=0, high=2, size=(batch_size, num_candidates, max_seq_len))
    model = VerbArgumentAttention(dim=dim, span_h=span_h, num_heads=num_heads, selection_h=selection_h, model_name="GroNLP/bert-base-dutch-cased")
    result = model(input_ids, input_masks, verb_tags, candidate_masks, candidate_tags)