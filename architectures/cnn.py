import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionLayer(nn.Module):
    def __init__(self, vocab_sz, embedding_dim, embedding_mat, filter_sizes=[2, 3, 4, 5], num_filters=20, hidden_dim=20):
        super(ConvolutionLayer, self).__init__()

        self.emb_layer = nn.Embedding(vocab_sz, embedding_dim)
        self.emb_layer.weights = torch.nn.Parameter(torch.from_numpy(embedding_mat))

        self.relu = nn.ReLU(True)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.dense_1 = nn.Linear(in_features=num_filters * len(filter_sizes), out_features=hidden_dim)
        self.dense_2 = nn.Linear(in_features=hidden_dim, out_features=4)
        self.softmax_layer = nn.LogSoftmax(dim=1)
        self.conv_blocks = nn.ModuleList()
        for sz in filter_sizes:
            conv = nn.Conv1d(100, num_filters, kernel_size=sz, stride=1, padding=0)
            self.conv_blocks.append(conv)


    def forward(self, x):
        x = self.emb_layer(x) # shape (batch, 10, 100))
        x = x.permute(0, 2, 1)
        output_list = []
        for conv_layer in self.conv_blocks:
            output = conv_layer(x)
            output = self.relu(output)
            output = self.max_pool(output).squeeze()
            output_list.append(output)
        x = torch.cat(output_list, dim=1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.softmax_layer(x)
        return x
