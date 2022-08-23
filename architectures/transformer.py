import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from torch.optim import Adam
import torch.optim as optim
from tqdm import tqdm

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
        self.softmax_layer = nn.LogSoftmax(dim=1)

        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.linear = nn.Linear(768, 4)
        # self.softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, input_id, mask):

        # _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        # linear_output = self.linear(pooled_output)
        # final_layer = self.softmax_layer(linear_output)

        output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)[0]
        final_layer = self.softmax_layer(output)

        return final_layer