import torch
import torch.nn as nn
import torch.nn.functional as F


# Attention layer for words and sentences
class AttentionWithContext(nn.Module):
    def __init__(self):
        super(AttentionWithContext, self).__init__()

        self.W = torch.nn.Parameter(torch.empty(100, 100))
        self.u = torch.nn.Parameter(torch.empty(100, ))
        self.b = torch.nn.Parameter(torch.empty(100, ))
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.normal_(self.u, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def forward(self, x, mask=None):
        uit = torch.matmul(x, self.W)

        uit += self.b

        uit = torch.tanh(uit)
        ait = torch.matmul(uit, self.u)

        a = torch.exp(ait)

        a = a / torch.sum(a, axis=1, keepdim=True) + 1e-7
        a = a.unsqueeze(-1)
        weighted_input = x * a

        return torch.sum(weighted_input, axis=1)


# Hierarchical Attention Encoder
class HierAttLayerEncoder(nn.Module):
    def __init__(self, vocab_sz, embedding_dim, embedding_mat):
        super(HierAttLayerEncoder, self).__init__()

        self.emb_layer = nn.Embedding(vocab_sz, embedding_dim)
        self.emb_layer.weights = torch.nn.Parameter(
            torch.from_numpy(embedding_mat))
        self.l_lstm = nn.GRU(input_size=100,
                             hidden_size=100,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True)
        # self.l_lstm = nn.GRU(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, batch_first=False)
        self.l_dense = nn.Linear(in_features=200, out_features=100)
        self.l_att = AttentionWithContext()

    def forward(self, x, h):
        embedded_sequences = self.emb_layer(x)
        out, h = self.l_lstm(embedded_sequences, h)
        out = self.l_dense(out)
        out = self.l_att(out)
        return out, h

class DataWrapper():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
