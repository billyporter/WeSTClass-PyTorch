import torch
import torch.nn as nn
import torch.nn.functional as F

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

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
        # hidden_size = 100
        # self.word_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.word_weight.data.normal_(0.0, 0.05)
        # self.word_bias = nn.Parameter(torch.Tensor(1, hidden_size))
        # self.context_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
        # self.context_weight.data.normal_(0.0, 0.05)
        

    # def forward(self, x):
    #     output = matrix_mul(x, self.word_weight, self.word_bias)
    #     output = matrix_mul(output, self.context_weight).permute(1,0)
    #     output = F.softmax(output)
    #     output = element_wise_mul(x,output.permute(1,0))
    #     print(output.shape)
    #     return output

    def forward(self, x, mask=None):
        # print(x.shape)
        # print().shape
        uit = torch.matmul(x, self.W)

        uit += self.b

        uit = torch.tanh(uit)
        ait = torch.matmul(uit, self.u)

        a = torch.exp(ait)

        a = a / torch.sum(a, axis=1, keepdim=True) + 1e-7
        a = a.unsqueeze(-1)
        weighted_input = x * a
        # print('asdfasdf')
        # print(torch.sum(weighted_input, axis=1).shape)
        # print().shape
        return torch.sum(weighted_input, axis=1)


# Hierarchical Attention Encoder
class HierAttLayerEncoder(nn.Module):
    def __init__(self, vocab_sz, embedding_dim, embedding_mat):
        super(HierAttLayerEncoder, self).__init__()

        self.emb_layer = nn.Embedding(vocab_sz, embedding_dim)
        self.emb_layer.weights = torch.nn.Parameter(torch.from_numpy(embedding_mat))
        self.l_lstm = nn.GRU(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        self.l_dense = TimeDistributed(nn.Linear(in_features=200, out_features=100))
        self.l_att = AttentionWithContext()

    def forward(self, x):
        print(x.shape)
        
        embedded_sequences = self.emb_layer(x)
        # print(embedded_sequences.shape)
        out, h = self.l_lstm(embedded_sequences)
        # print(out.shape)
        out = self.l_dense(out)
        out = self.l_att(out)
        return out, h

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
