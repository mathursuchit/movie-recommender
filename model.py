import torch
import torch.nn as nn


class MF(nn.Module):
    """
    Matrix Factorization — learns a vector for each user and each movie.
    Predicted rating = dot product of user and movie vectors.

    Unlike NCF, item embeddings here are pure item representations —
    not conditioned on users — so cosine similarity between them works well
    for finding similar movies at inference time.
    """

    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        dot = (u * v).sum(dim=1, keepdim=True)
        bias = self.user_bias(user_ids) + self.item_bias(item_ids)
        return torch.sigmoid(dot + bias).squeeze()

    def get_item_embeddings(self):
        return self.item_emb.weight.detach()
