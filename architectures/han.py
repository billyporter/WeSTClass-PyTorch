import torch
import torch.nn as nn
import torch.nn.functional as F


class HierAttLayer(nn.Module):
    def __init__(self, vocab_sz, embedding_dim, embedding_mat):
        super(HierAttLayer, self).__init__()

        # Word level encoder
        self.encoder = HierAttLayerEncoder(vocab_sz, embedding_dim, embedding_mat)

        # Sentence level encoder
        self.l_lstm_sent = nn.GRU(input_size=100,
                                  hidden_size=100,
                                  num_layers=2,
                                  bidirectional=True,
                                  batch_first=True)
        self.l_dense_sent = nn.Linear(in_features=200, out_features=100)
        self.l_att_sent = AttentionWithContext()
        
        # Class predictions
        self.l_linear = nn.Linear(in_features=100, out_features=4)
        self.softmax_layer = nn.LogSoftmax(dim=1)
        
        # LSTM Hidden State
        self._init_hidden_state(0)

    def forward(self, x):
        output_list = []
        
        x = x.permute(1, 0, 2)
        for i in x:
            output, self.sent_hidden_state = self.encoder(i, self.sent_hidden_state)
            output_list.append(output)

        x = torch.stack(output_list)
        x = x.permute(1, 0, 2)
        x, h = self.l_lstm_sent(x)
        x = self.l_dense_sent(x)
        x = self.l_att_sent(x)
        x = self.l_linear(x)
        x = self.softmax_layer(x)
        return x

    def _init_hidden_state(self, batch_size):
        self.sent_hidden_state = None
        # pass
        # self.sent_hidden_state = torch.zeros(2 * 2, 10, 100)
        # if torch.cuda.is_available():
        #     self.sent_hidden_state = self.sent_hidden_state.cuda()

# Hierarchical Attention Encoder
class HierAttLayerEncoder(nn.Module):
    def __init__(self, vocab_sz, embedding_dim, embedding_mat):
        super(HierAttLayerEncoder, self).__init__()

        weights = torch.from_numpy(embedding_mat).type(torch.FloatTensor)
        self.emb_layer = nn.Embedding(vocab_sz, embedding_dim).from_pretrained(weights)
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
