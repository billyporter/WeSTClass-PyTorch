import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from layers import binary_accuracy
from torch.optim import Adam
import torch.optim as optim
from tqdm import tqdm

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        # self.relu = nn.ReLU()
        self.softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        # print(pooled_output)
        # dropout_output = self.dropout(pooled_output)
        # print(dropout_output)
        linear_output = self.linear(pooled_output)
        # print(linear_output)
        final_layer = self.softmax_layer(linear_output)

        return final_layer


def tokenize_bert(seed_documents, tokenizer):
    seed_text = []
    for doc in seed_documents:
        local_str = ''
        for i, sent in enumerate(doc):
            local_str += ' '.join(sent[0:15])
            if i == 2:
                break
        seed_text.append(local_str)

    texts = [tokenizer(text,padding='max_length', max_length = 128, 
                       truncation=True, return_tensors="pt") for text in seed_text]

    return texts


def train_bert(model, train_loader):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model = model.cuda()

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = Adam(model.parameters(), lr=0.0001)


    for epoch in range(10):

        train_correct = 0
        train_loss = 0

        for train_input, train_label in tqdm(train_loader):
            # print()
            input_id = train_input['input_ids'].squeeze(1)
            mask = train_input['attention_mask']
            
            if is_cuda:
                input_id = input_id.cuda()
                mask = mask.cuda()
                train_label = train_label.cuda()

            output = model(input_id, mask)


            train_correct += binary_accuracy(output.cpu().detach(), train_label.cpu().detach())

            batch_loss = criterion(output, train_label)
            train_loss += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print('Epoch ({}) Train accuracy: {}'.format(epoch, train_correct / len(train_loader.dataset)))
        print('Epoch ({}) Train loss: {}'.format(epoch, train_loss))