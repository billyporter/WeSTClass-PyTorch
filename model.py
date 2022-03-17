import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import f1_score
from time import time
import copy, random

class DataWrapper():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class AttentionWithContext(nn.Module):
    def __init__(self):
        super(AttentionWithContext, self).__init__()

        self.W = torch.nn.Parameter(torch.empty(100, 100))
        self.u = torch.nn.Parameter(torch.empty(100, ))
        torch.nn.init.normal_(self.W, std=0.01)
        torch.nn.init.normal_(self.u, std=0.01)

    def forward(self, x, mask=None):
        uit = torch.matmul(x, self.W)

        uit = torch.tanh(uit)
        ait = torch.matmul(uit, self.u)

        a = torch.exp(ait)

        a = a / torch.sum(a, axis=1, keepdim=True) + 1e-7
        a = a.unsqueeze(-1)
        weighted_input = x * a

        return torch.sum(weighted_input, axis=1)
        

class HierAttLayerEncoder(nn.Module):
    def __init__(self, vocab_sz, embedding_dim, embedding_mat):
        super(HierAttLayerEncoder, self).__init__()

        self.emb_layer = nn.Embedding(vocab_sz, embedding_dim)
        self.emb_layer.weights = torch.nn.Parameter(torch.from_numpy(embedding_mat))
        self.l_lstm = nn.GRU(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        self.l_dense = TimeDistributed(nn.Linear(in_features=200, out_features=100))
        self.l_att = AttentionWithContext()

    def forward(self, x):
        embedded_sequences = self.emb_layer(x)
        out, h = self.l_lstm(embedded_sequences)
        out = self.l_dense(out)
        out = self.l_att(out)
        return out


class HierAttLayer(nn.Module):
    def __init__(self, vocab_sz, embedding_dim, embedding_mat):
        super(HierAttLayer, self).__init__()
        self.encoder = HierAttLayerEncoder(vocab_sz, embedding_dim, embedding_mat)

        # Decoder
        self.review_encoder = TimeDistributed(self.encoder)
        self.l_lstm_sent = nn.GRU(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        self.l_dense_sent = TimeDistributed(nn.Linear(in_features=200, out_features=100))
        self.l_att_sent = AttentionWithContext()
        self.l_linear = nn.Linear(in_features=100, out_features=4)
        self.softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.squeeze()
        x = self.review_encoder(x)
        x = x.unsqueeze(0)
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
                 batch_size = 256,
                 classifier = None
                 ):
        
        self.batch_size = batch_size
        self.classifier = HierAttLayer(vocab_sz, word_embedding_dim, embedding_matrix)
        self.is_cuda = torch.cuda.is_available()
        if classifier != None:
            self.classifier = classifier
        if self.is_cuda:
            self.classifier = self.classifier.cuda()

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = self.classifier
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
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

            # here i detach it, it is very likely that i might need to be tensors
            q_list.append(torch.exp(pred).cpu().detach().numpy()[0])
            guess = torch.argmax(pred.cpu())
            preds_list.append(guess)
        return q_list, preds_list

    def test_load_deprecated(self, test_loader):
        test_total = 0
        test_correct = 0
        for i, (document, label) in enumerate(test_loader):
            test_total += 1
            if self.is_cuda:
                document = document.cuda()
                # label = label.cuda()
            # document = Variable(document)
            print(document.shape)

            feature = self.classifier(document)

            # Get categorical target
            # print(feature)
            # print(label)
            print().shape
            # num = torch.argmax(label.cpu())
            guess = torch.argmax(feature.cpu())
            num = label[0]
            # print(num)
            if guess == num:
                test_correct += 1


        print('Test accuracy: {}'.format(test_correct / test_total))

    def target_distribution(self, q, power=2):
        # square each class, divide by total of class
        weight = q ** power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        return p

    def pretrain(self, train_loader):
        epochs = 10
        pretrain_output_file = open('pretrain_output.txt', 'w')
        t0 = time()
        best_dev_loss = None
        print('\nPretraining...')
        print('\nPretraining...', file=pretrain_output_file)
        for epoch in range(epochs):
            print('------EPOCH: ' + str(epoch) + '-------')
            print('------EPOCH: ' + str(epoch) + '-------', file=pretrain_output_file)
            train_loss = 0.
            train_correct = 0
            train_total = 0
            loss_list = []
            for i, (document, label) in enumerate(tqdm(train_loader)):
                train_total += 1
                if self.is_cuda:
                    document = document.cuda()
                    label = label.cuda()
                document = Variable(document)


                feature = self.classifier(document)

                # Get categorical target
                num = torch.argmax(label.cpu())
                guess = torch.argmax(feature.cpu())
                if guess == num:
                    train_correct += 1


                # Compute Loss
                local_loss = self.criterion(feature, label)
                loss_list.append(local_loss)

                if (i + 1) % 256 == 0 or (i + 1) == 2000:

                    loss = sum(loss_list)
                    loss_list = []

                    # Clear gradient in optimizer 
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Do one step of gradient descent
                    self.optimizer.step()
                    train_loss += loss.item()


            print('Epoch ({}) Train accuracy: {}'.format(epoch, train_correct / train_total))
            print('Epoch ({}) Train loss: {}'.format(epoch, train_loss))
            print('Epoch ({}) Train accuracy: {}'.format(epoch, train_correct / train_total), file=pretrain_output_file)
            print('Epoch ({}) Train loss: {}'.format(epoch, train_loss), file=pretrain_output_file)


            if best_dev_loss is None or train_loss < best_dev_loss:
                print('Saving...')
                torch.save(self.classifier, "model")
                best_dev_loss = train_loss


        # Close output file
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        print('Pretraining time: {:.2f}s'.format(time() - t0), file=pretrain_output_file)
        pretrain_output_file.close()

    
    def self_train(self, train_loader, x, y=None, maxiter=500, tol=0.1, power=2,
                   update_interval=50, batch_size=256):
        print('a')
        pred, y_preds = self.predict(train_loader)
        print('b')
        y_preds = np.asarray(y_preds)
        y_preds_last = np.copy(y_preds)

        # Open file
        selftrain_file = open('selftrain_output.txt', 'w')
        t0 = time()

        index = 0
        index_array = np.arange(x.shape[0])
        for ite in tqdm(range(int(maxiter))):
            if ite % update_interval == 0:
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
                delta_label = np.sum(y_preds != y_preds_last).astype(float) / y_preds.shape[0]
                y_preds_last = np.copy(y_preds)
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label*100, 3)))
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label*100, 3)), file=selftrain_file)
                if ite > 0 and delta_label < tol/100:
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label*100, 3), tol))
                    print('Reached tolerance threshold. Stopping training.')
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label*100, 3), tol), file=selftrain_file)
                    print('Reached tolerance threshold. Stopping training.', file=selftrain_file)
                    print('Saving...')
                    torch.save(self.classifier, "finetuned_model")
                    break

            # Train on a singular batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]

            self.train_on_batch(x=x[idx], y=p[idx])

            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        # Close output file
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        print('Pretraining time: {:.2f}s'.format(time() - t0), file=selftrain_file)
        selftrain_file.close()

    def train_on_batch(self, x, y):
        batch_data = DataWrapper(x, y)
        batch_train_loader = DataLoader(dataset=batch_data, batch_size=1, shuffle=False)
        train_loss = 0.
        train_correct = 0
        train_total = 0
        loss_list = []
        for i, (document, label) in enumerate(batch_train_loader):
            train_total += 1
            if self.is_cuda:
                document = document.cuda()
                label = label.cuda()
            document = Variable(document)


            feature = self.classifier(document)

            # Get categorical target
            num = torch.argmax(label.cpu())
            guess = torch.argmax(feature.cpu())
            if guess == num:
                train_correct += 1


            # Compute Loss
            loss_list.append(self.criterion(feature, label))

        loss = sum(loss_list)
        loss_list = []

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