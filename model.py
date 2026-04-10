import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Collaborative Filtering (He et al., 2017)

    Each user and movie gets a learned embedding vector.
    Concatenate the two, run through MLP layers, output a score.
    """

    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32]):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        self.fc_layers = nn.ModuleList()
        input_size = embedding_dim * 2  # user + item concatenated
        for size in layers:
            self.fc_layers.append(nn.Linear(input_size, size))
            input_size = size

        self.output_layer = nn.Linear(layers[-1], 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for layer in self.fc_layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        x = torch.cat([u, v], dim=-1)
        for layer in self.fc_layers:
            x = self.relu(layer(x))
        return self.sigmoid(self.output_layer(x)).squeeze()

    def get_item_embeddings(self):
        # used at inference time to find similar movies
        return self.item_emb.weight.detach()
