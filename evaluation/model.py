"""Define the PyTorch model (using transformers and all)"""
import torch
from transformers import AutoModel


class SpanClassModel(torch.nn.Module):
    def __init__(self, dim: int, model_name: str, freeze: bool = True):
        """

        :type dim: int
        """
        super(SpanClassModel, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        if self.freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
        self.mlp1 = torch.nn.Linear(2*dim, 64)
        self.mlp2 = torch.nn.Linear(64, 32)
        self.layer_out = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()
        # self.softmax = torch.nn.Softmax()

    @staticmethod
    def extract_span_avg(embeds, tags):
        """

        :param embeds: [batch_size, max_seq_len, dims]
        :param tags: [batch_size, max_seq_len]
        :return: the average of embeds: [batch_size, dims]
        """
        return torch.sum(tags.unsqueeze(dim=2) * embeds, axis=1) / torch.sum(tags.unsqueeze(dim=2), axis=1)

    def forward(self, input_ids, input_masks, input_subj_tags, input_obj_tags):
        embeddings = self.bert_model(input_ids, attention_mask=input_masks)[0]
        subj_span_embedding = self.extract_span_avg(embeddings, input_subj_tags)
        obj_span_embedding = self.extract_span_avg(embeddings, input_obj_tags)
        mlp_1 = self.relu(self.mlp1(torch.cat([subj_span_embedding, obj_span_embedding], axis=1)))
        mlp_2 = self.relu(self.mlp2(mlp_1))
        out = self.layer_out(mlp_2)
        # out = self.layer_out(mlp_2).view(-1)
        # out = self.softmax(self.softmax_out(mlp_2))
        return out


class SpanClassModelAttention(torch.nn.Module):
    def __init__(self, dim: int, model_name: str, freeze: bool = True):
        """

        :type dim: int
        """
        super(SpanClassModelAttention, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        if self.freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        self.mlp1 = torch.nn.Linear(2*dim, 64)
        self.mlp2 = torch.nn.Linear(64, 32)
        self.layer_out = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()
        # self.softmax = torch.nn.Softmax()

    def extract_span_embedding(self, embeds, tags):
        """

        :param embeds: [batch_size, max_seq_len, dims]
        :param tags: [batch_size, max_seq_len]
        :return: the average of embeds: [batch_size, dims]
        """
        inv_tags = 1 - tags
        attn_output, attn_weights = self.self_attn(query=embeds.transpose(1, 0), key=embeds.transpose(1, 0),
                                                   value=embeds.transpose(1, 0), key_padding_mask=inv_tags)
        import pdb
        pdb.set_trace()
        return torch.sum(torch.matmul(embeds.transpose(1, 2), attn_weights).transpose(1, 2), axis=1)

""

    def forward(self, input_ids, input_masks, input_subj_tags, input_obj_tags):
        """
            :param input_ids: [batch_size, max_seq_len]
            :param input_masks: [batch_size, max_seq_len]
            :param input_subj_tags: [batch_size, max_seq_len]
            :param input_obj_tags: [batch_size, max_seq_len]
            :return: a binary classification embedding: [batch_size, 2]
        """
        embeddings = self.bert_model(input_ids, attention_mask=input_masks)[0]
        subj_span_embedding = self.extract_span_embedding(embeddings, input_subj_tags)
        obj_span_embedding = self.extract_span_embedding(embeddings, input_obj_tags)
        mlp_1 = self.relu(self.mlp1(torch.cat([subj_span_embedding, obj_span_embedding], axis=1)))
        mlp_2 = self.relu(self.mlp2(mlp_1))
        out = self.layer_out(mlp_2)
        # out = self.layer_out(mlp_2).view(-1)
        # out = self.softmax(self.softmax_out(mlp_2))
        return out

class SpanClassModelAttentionN(torch.nn.Module):
    def __init__(self, dim: int, spans: int, model_name: str, freeze: bool = True):
        """

        :type dim: int
        """
        super(SpanClassModelAttentionN, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        if self.freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        self.spans = spans
        self.mlp1 = torch.nn.Linear(spans*dim, 64)
        self.mlp2 = torch.nn.Linear(64, 32)
        self.layer_out = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()
        # self.softmax = torch.nn.Softmax()

    def extract_span_embedding(self, embeds, tags):
        """

        :param embeds: [batch_size, max_seq_len, dims]
        :param tags: [batch_size, max_seq_len]
        :return: the average of embeds: [batch_size, dims]
        """
        inv_tags = 1 - tags
        attn_output, attn_weights = self.self_attn(query=embeds.transpose(1, 0), key=embeds.transpose(1, 0),
                                                   value=embeds.transpose(1, 0), key_padding_mask=inv_tags)
        return torch.sum(torch.matmul(embeds.transpose(1, 2), attn_weights).transpose(1, 2), axis=1)


    def forward(self, input_ids, input_masks, input_tags):
        """
            :param input_ids: [batch_size, max_seq_len]
            :param input_masks: [batch_size, max_seq_len]
            :param input_tags: [spans, batch_size, max_seq_len]
            :return: a binary classification embedding: [batch_size, 2]
        """
        embeddings = self.bert_model(input_ids, attention_mask=input_masks)[0]
        span_embeddings = [self.extract_span_embedding(embeddings, tags) for tags in input_tags]
        mlp_1 = self.relu(self.mlp1(torch.cat(span_embeddings, axis=1)))
        mlp_2 = self.relu(self.mlp2(mlp_1))
        out = self.layer_out(mlp_2)
        return out

class SpanClassModelAttentionBatched(torch.nn.Module):
    def __init__(self, dim: int, model_name: str, freeze: bool = True):
        """

        :type dim: int
        """
        super(SpanClassModelAttentionBatched, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        if self.freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        self.mlp1 = torch.nn.Linear(2*dim, 64)
        self.mlp2 = torch.nn.Linear(64, 32)
        self.layer_out = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()
        # self.softmax = torch.nn.Softmax()

    def extract_span_embedding(self, embeds, tags):
        """

        :param embeds: [batch_size, max_seq_len, dims]
        :param tags: [batch_size, max_seq_len]
        :return: the average of embeds: [batch_size, dims]
        """
        inv_tags = 1 - tags
        attn_output, attn_weights = self.self_attn(query=embeds.transpose(1, 0), key=embeds.transpose(1, 0),
                                                   value=embeds.transpose(1, 0), key_padding_mask=inv_tags)
        return torch.sum(torch.matmul(embeds.transpose(1, 2), attn_weights).transpose(1, 2), axis=1)

    def forward(self, input_ids, input_masks, input_arg_tags):
        """
            :param input_ids: [batch_size, max_seq_len]
            :param input_masks: [batch_size, max_seq_len]
            :param input_arg_tags: [batch_size, max_seq_len]

            :param input_subj_tags: [batch_size, max_seq_len]
            :param input_obj_tags: [batch_size, max_seq_len]
            :return: a binary classification embedding: [batch_size, 2]
        """
        embeddings = self.bert_model(input_ids, attention_mask=input_masks)[0]
        subj_span_embedding = self.extract_span_embedding(embeddings, input_subj_tags)
        obj_span_embedding = self.extract_span_embedding(embeddings, input_obj_tags)
        mlp_1 = self.relu(self.mlp1(torch.cat([subj_span_embedding, obj_span_embedding], axis=1)))
        mlp_2 = self.relu(self.mlp2(mlp_1))
        out = self.layer_out(mlp_2)
        # out = self.layer_out(mlp_2).view(-1)
        # out = self.softmax(self.softmax_out(mlp_2))
        return out


class SpanClassModelAttentionVerb(torch.nn.Module):
    def __init__(self, dim: int, model_name: str, freeze: bool = True):
        """

        :type dim: int
        """
        super(SpanClassModelAttentionVerb, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        if self.freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        self.mlp1 = torch.nn.Linear(dim, 64)
        self.mlp2 = torch.nn.Linear(64, 32)
        self.layer_out = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()
        # self.softmax = torch.nn.Softmax()

    def extract_span_embedding(self, embeds, tags):
        """

        :param embeds: [batch_size, max_seq_len, dims]
        :param tags: [batch_size, max_seq_len]
        :return: the average of embeds: [batch_size, dims]
        """
        inv_tags = 1 - tags
        attn_output, attn_weights = self.self_attn(query=embeds.transpose(1, 0), key=embeds.transpose(1, 0),
                                                   value=embeds.transpose(1, 0), key_padding_mask=inv_tags)
        return torch.sum(torch.matmul(embeds.transpose(1, 2), attn_weights).transpose(1, 2), axis=1)

    def forward(self, input_ids, input_masks, input_verb_tags):
        embeddings = self.bert_model(input_ids, attention_mask=input_masks)[0]
        verb_span_embedding = self.extract_span_embedding(embeddings, input_verb_tags)
        mlp_1 = self.relu(self.mlp1(verb_span_embedding))
        mlp_2 = self.relu(self.mlp2(mlp_1))
        out = self.layer_out(mlp_2)
        # out = self.layer_out(mlp_2).view(-1)
        # out = self.softmax(self.softmax_out(mlp_2))
        return out


