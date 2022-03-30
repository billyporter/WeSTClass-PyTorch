import torch
import torch.nn as nn


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
        # self.emb_layer.weights = torch.nn.Parameter(torch.from_numpy(embedding_mat))
        self.l_lstm = nn.GRU(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        self.l_dense = TimeDistributed(nn.Linear(in_features=200, out_features=100))
        self.l_att = AttentionWithContext()

    def forward(self, x):
        embedded_sequences = self.emb_layer(x)
        out, h = self.l_lstm(embedded_sequences)
        out = self.l_dense(out)
        out = self.l_att(out)
        return out

# Allows to apply a layer to every temporal slice of an input
# Equivalent of Keras https://keras.io/api/layers/recurrent_layers/time_distributed/
# Source: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class DataWrapper():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
