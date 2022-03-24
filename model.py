import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import f1_score
from time import time
import copy, random
from layers import *


class HierAttLayer(nn.Module):
    def __init__(self, vocab_sz, embedding_dim, embedding_mat):
        super(HierAttLayer, self).__init__()

        # Encoder
        self.encoder = HierAttLayerEncoder(vocab_sz, embedding_dim,
                                           embedding_mat)

        # Decoder
        self.review_encoder = TimeDistributed(TimeDistributed(self.encoder))
        self.l_lstm_sent = nn.GRU(input_size=100,
                                  hidden_size=100,
                                  num_layers=2,
                                  bidirectional=True,
                                  batch_first=True)
        self.l_dense_sent = TimeDistributed(
            nn.Linear(in_features=200, out_features=100))
        self.l_att_sent = AttentionWithContext()
        self.l_linear = nn.Linear(in_features=100, out_features=4)
        self.softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.review_encoder(x)
        x, h = self.l_lstm_sent(x)
        x = self.l_dense_sent(x)
        x = self.l_att_sent(x)
        x = self.l_linear(x)
        x = self.softmax_layer(x)
        return x

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data
        hidden = weight.new(2, batch_size, 100).zero_()
        return hidden


class WSTC():
    def __init__(self,
                 input_shape,
                 n_classes=None,
                 model='rnn',
                 vocab_sz=None,
                 word_embedding_dim=100,
                 embedding_matrix=None,
                 batch_size=256,
                 classifier=None):

        self.batch_size = batch_size
        self.classifier = HierAttLayer(vocab_sz, word_embedding_dim,
                                       embedding_matrix)
        self.is_cuda = torch.cuda.is_available()
        if classifier != None:
            self.classifier = classifier
        if self.is_cuda:
            self.classifier = self.classifier.cuda()

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = self.classifier
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.01)
        # self.optimizer = optim.SGD(self.classifier.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def predict(self, x):
        # TODO: Change to numpy objects
        q_list = []
        preds_list = []
        for document in x:
            document = document[0]
            if self.is_cuda:
                document = document.cuda()

            # document = Variable(document)
            pred = self.model(document)
            preds_numpy = torch.exp(pred).cpu().detach().numpy()
            guesses = torch.argmax(pred.cpu(), dim=1)
            q_list.extend(preds_numpy)
            preds_list.extend(guesses)

        return q_list, preds_list

    def evaluate_dataset(self, test_loader, get_stats=False):
        test_correct = 0
        confidence_list = []
        preds_list = []
        actual_list = []
        for i, (document, label) in enumerate(tqdm(test_loader)):
            if self.is_cuda:
                document = document.cuda()

            feature = self.classifier(document)
            # Get categorical target
            if get_stats:
                scores, indices = confidence_correct(torch.exp(feature).cpu().detach(), label.cpu().detach())
                scores = scores.tolist()
                indices = indices.tolist()
                confidence_list.extend(scores)
                preds_list.extend(indices)
                actual_list.extend(label.tolist())
            
            test_correct += binary_accuracy(feature.cpu().detach(), label.cpu().detach(), method="eval")

        print('Test accuracy: {}'.format(test_correct / len(test_loader.dataset)))

        # Write confdience and booleans to file
        if get_stats:
            np.save("confidence_array.npy", np.asarray(confidence_list))
            np.save("preds_array.npy", np.asarray(preds_list))
            np.save("actual_array.npy", np.asarray(actual_list))


    def target_distribution(self, q, power=2):
        # square each class, divide by total of class
        weight = q**power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        return p

    def pretrain(self,
                 train_loader,
                 epochs,
                 output_save_path='pretrain_output.txt',
                 model_save_path="model.pt"):
        pretrain_output_file = open(output_save_path, 'w')
        t0 = time()
        best_dev_loss = None
        print('\nPretraining...')
        print('\nPretraining...', file=pretrain_output_file)
        for epoch in range(epochs):
            print('------EPOCH: ' + str(epoch) + '-------')
            print('------EPOCH: ' + str(epoch) + '-------',
                  file=pretrain_output_file)
            train_loss = 0.
            train_correct = 0
            for i, (document, label) in enumerate(tqdm(train_loader)):
                if self.is_cuda:
                    document = document.cuda()
                    label = label.cuda()
                document = Variable(document)

                feature = self.classifier(document)

                # Get categorical target
                train_correct += binary_accuracy(feature.cpu().detach(), label.cpu().detach(), method="train")

                # Compute Loss
                loss = self.criterion(feature, label)

                # Clear gradient in optimizer
                self.optimizer.zero_grad()
                loss.backward()

                # Do one step of gradient descent
                self.optimizer.step()
                train_loss += loss.item()

            print('Epoch ({}) Train accuracy: {}'.format(epoch, train_correct / len(train_loader.dataset)))
            print('Epoch ({}) Train accuracy: {}'.format(epoch, train_correct / len(train_loader.dataset)), file=pretrain_output_file)
            print('Epoch ({}) Train loss: {}'.format(epoch, train_loss))
            print('Epoch ({}) Train loss: {}'.format(epoch, train_loss), file=pretrain_output_file)

            if best_dev_loss is None or train_loss < best_dev_loss:
                print('Saving...')
                torch.save(self.classifier, model_save_path)
                best_dev_loss = train_loss

        # Close output file
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        print('Pretraining time: {:.2f}s'.format(time() - t0),
              file=pretrain_output_file)
        pretrain_output_file.close()

    def self_train(self,
                   train_loader,
                   x,
                   y=None,
                   maxiter=500,
                   tol=0.1,
                   power=2,
                   update_interval=100,
                   output_save_path='selftrain_output.txt',
                   model_save_path="finetuned_model.pt"):

        q, y_preds = self.predict(train_loader)
        y_preds = np.asarray(y_preds)
        y_preds_last = np.copy(y_preds)

        # Open file
        selftrain_file = open(output_save_path, 'w')
        t0 = time()

        index = 0
        index_array = np.arange(x.shape[0])
        for ite in tqdm(range(int(maxiter))):
            if ite % update_interval == 0:
                if ite != 0:
                    q, y_preds = self.predict(train_loader)

                y_preds = np.asarray(y_preds)
                q = np.asarray(q)
                p = self.target_distribution(q)
                print('\nIter {}: '.format(ite), end='')
                print('\nIter {}: '.format(ite), end='', file=selftrain_file)
                if y is not None:
                    f1_macro, f1_micro = np.round(f1(y, y_preds), 5)
                    print('f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro))
                    print('f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro), file=selftrain_file)
                    # test_load_deprecated()

                # Check stop criterion
                delta_label = np.sum(
                    y_preds != y_preds_last).astype(float) / y_preds.shape[0]
                y_preds_last = np.copy(y_preds)
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label * 100, 3)))
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label * 100, 3)), file=selftrain_file)
                if ite > 0 and delta_label < tol / 100:
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label * 100, 3), tol))
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label * 100, 3), tol),file=selftrain_file)
                    print('Reached tolerance threshold. Stopping training.')
                    print('Reached tolerance threshold. Stopping training.',file=selftrain_file)
                    print('Saving...')
                    torch.save(self.classifier, model_save_path)
                    break

            # Train on a singular batch
            idx = index_array[index * self.batch_size:min((index + 1) * self.batch_size, x.shape[0])]

            self.train_on_batch(x=x[idx], y=p[idx], batch_size=self.batch_size)

            index = index + 1 if (index + 1) * self.batch_size <= x.shape[0] else 0

        # Close output file
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        print('Pretraining time: {:.2f}s'.format(time() - t0), file=selftrain_file)
        selftrain_file.close()

    def train_on_batch(self, x, y, batch_size):
        batch_data = DataWrapper(x, y)
        batch_train_loader = DataLoader(dataset=batch_data,
                                        batch_size=batch_size,
                                        shuffle=False)
        train_loss = 0.
        train_correct = 0
        for i, (document, label) in enumerate(batch_train_loader):
            if self.is_cuda:
                document = document.cuda()
                label = label.cuda()
            document = Variable(document)

            feature = self.classifier(document)

            # Get categorical target
            train_correct += binary_accuracy(feature.cpu().detach(), label.cpu().detach())

            # Compute Loss
            loss = self.criterion(feature, label)

            # Clear gradient in optimizer
            self.optimizer.zero_grad()
            loss.backward()

            # Do one step of gradient descent
            self.optimizer.step()
            train_loss += loss.item()


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