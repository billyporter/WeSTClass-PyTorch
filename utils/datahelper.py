import torch
from sklearn.metrics import f1_score
import numpy as np


class DataWrapper():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class BertDataWrapper():
    def __init__(self, documents, y_data):
        self.input_ids = documents["input_ids"]
        self.attention_mask = documents["attention_mask"]
        self.y = y_data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.input_ids[index], self.attention_mask[index]], self.y[index]
    
    
def f1(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    return f1_macro, f1_micro


def binary_accuracy(preds, y, method="train"):
    preds = torch.argmax(preds, dim=1)
    if method == "train":
        y = torch.argmax(y, dim=1)
    count = torch.sum(preds == y)
    return count


def confidence_correct(preds, y, method="eval"):
    max_value = torch.max(preds, dim=1)
    score, indices = max_value
    return score, indices
