import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from time import time
from utils.datahelper import *
from architectures.han import *
from architectures.cnn import *
from architectures.transformer import *
from sklearn.metrics import f1_score


class WSTC():
    def __init__(self,
                 n_classes=4,
                 model='rnn',
                 vocab_sz=67765,
                 word_embedding_dim=100,
                 embedding_matrix=None,
                 batch_size=256,
                 learning_rate=0.01,
                 classifier=None,
                 sup_source="keywords"):

        self.batch_size = batch_size
        self.sup_source = sup_source

        if model == 'rnn':
            self.classifier = HierAttLayer(vocab_sz, word_embedding_dim,
                                       embedding_matrix)
        elif model == 'cnn':
            self.classifier = ConvolutionLayer(vocab_sz, word_embedding_dim,
                                       embedding_matrix)
        elif model == 'bert':
            self.classifier = BertClassifier()

        self.model = model
        self.is_cuda = torch.cuda.is_available()
        if classifier != None:
            self.classifier = classifier
        if self.is_cuda:
            self.classifier = self.classifier.cuda()
        self.sup_list = {}
        self.n_classes = n_classes
        # self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        # self.optimizer  = torch.optim.SGD(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=0.01, momentum=0.9)
        if self.model in ('rnn', 'bert'):
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.classifier.parameters(), lr=learning_rate, momentum=0.9)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.lr = learning_rate

    ### Generates predictions and class scores ###
    def predict(self, x):
        q_list = []
        preds_list = []
        for document in x:
            document = document[0]
            if self.model=='rnn':
                with torch.no_grad():
                    self.classifier._init_hidden_state(0)
                if self.is_cuda:
                    document = document.cuda()
                pred = self.classifier(document)
            elif self.model == 'cnn':
                if self.is_cuda:
                    document = document.cuda()
                document = Variable(document)
                pred = self.classifier(document)
            elif self.model=='bert':
                input_id, mask = document
                if self.is_cuda:
                    input_id = input_id.cuda()
                    mask = mask.cuda()
                pred = self.classifier(input_id, mask)

            preds_numpy = torch.exp(pred).cpu().detach().numpy()
            guesses = torch.argmax(pred.cpu(), dim=1)
            q_list.extend(preds_numpy)
            preds_list.extend(guesses)
        return q_list, preds_list

    ### Scores accuracy of model ###
    def evaluate_dataset(self, test_loader, get_stats=False, get_other=True):
        test_correct = 0
        confidence_list = []
        preds_list = []
        actual_list = []
        preds_map = {}
        for i, (document, label) in enumerate(tqdm(test_loader)):
            if self.model=='rnn':
                with torch.no_grad():
                    self.classifier._init_hidden_state(0)
                if self.is_cuda:
                    document = document.cuda()
                    label = label.cuda()
                feature = self.classifier(document)
            elif self.model == 'cnn':
                if self.is_cuda:
                    document = document.cuda()
                    label = label.cuda()
                feature = self.classifier(document)
            elif self.model == 'bert':
                input_id, mask = document
                if self.is_cuda:
                    input_id = input_id.cuda()
                    mask = mask.cuda()
                    label = label.cuda()
                feature = self.classifier(input_id, mask)
            
            # get_stats: get data for precision / recall curves
            if get_stats:
                scores, indices = confidence_correct(torch.exp(feature).cpu().detach(), label.cpu().detach())
                scores = scores.tolist()
                indices = indices.tolist()
                confidence_list.extend(scores)
                preds_list.extend(indices)
                actual_list.extend(label.tolist())

            # get_other: get number of predictions for each class
            if get_other:
                scores, indices = confidence_correct(torch.exp(feature).cpu().detach(), label.cpu().detach())
                indices = indices.tolist()
                for index in indices:
                    preds_map[index] = preds_map.get(index, 0) + 1

            preds_list.extend(indices)
            actual_list.extend(label.tolist())

            test_correct += binary_accuracy(feature.cpu().detach(), label.cpu().detach(), method="eval")

        print('Test accuracy: {}'.format(test_correct / len(test_loader.dataset)))
        f1_macro = np.round(f1_score(actual_list, preds_list, average='macro'), 5)
        f1_micro = np.round(f1_score(actual_list, preds_list, average='micro'), 5)
        print('f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro))

        # Write confidence and booleans to file
        if get_stats:
            np.save("confidence_array.npy", np.asarray(confidence_list))
            np.save("preds_array.npy", np.asarray(preds_list))
            np.save("actual_array.npy", np.asarray(actual_list))

    def pretrain(self, train_loader, epochs, sup_idx=None, output_save_path='pretrain_output.txt', model_save_path="model.pt"):
        pretrain_output_file = open(output_save_path, 'w')
        print("Epochs: {}, Batch size: {}, LR: {}".format(epochs, self.batch_size, self.lr), file=pretrain_output_file)
        output_save_path = "{}_{}_pretrain_output.txt".format(self.model, self.sup_source)
        model_save_path = "{}_model.pt".format(self.model)
        t0 = time()
        best_dev_loss = None
        print('\nPretraining...')
        print('\nPretraining...', file=pretrain_output_file)

        # For docs use only
        if sup_idx is not None:
            for i, seed_idx in enumerate(sup_idx):
                for idx in seed_idx:
                    self.sup_list[idx] = i
        
        for epoch in range(epochs):
            print('------EPOCH: ' + str(epoch) + '-------')
            print('------EPOCH: ' + str(epoch) + '-------', file=pretrain_output_file)
            train_loss = 0.
            train_correct = 0
            for i, (document, label) in enumerate(tqdm(train_loader)):
                if self.model == 'rnn':
                    self.classifier._init_hidden_state(self.batch_size)
                    if self.is_cuda:
                        document = document.cuda()
                        label = label.cuda()
                    document = Variable(document)
                    feature = self.classifier(document)

                elif self.model == 'cnn':
                    if self.is_cuda:
                        document = document.cuda()
                        label = label.cuda()
                    document = Variable(document)
                    feature = self.classifier(document)

                elif self.model == 'bert':
                    input_id, mask = document
                    if self.is_cuda:
                        input_id = input_id.cuda()
                        mask = mask.cuda()
                        label = label.cuda()
                    feature = self.classifier(input_id, mask)

                # Get batch accuracy
                train_correct += binary_accuracy(feature.cpu().detach(), label.cpu().detach(), method="train")
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
                # torch.save(self.classifier, model_save_path)
                best_dev_loss = train_loss

        # Close output file
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        print('Pretraining time: {:.2f}s'.format(time() - t0), file=pretrain_output_file)
        pretrain_output_file.close()

    def self_train(self,
                   train_loader,
                   x,
                   y=None,
                   learning_rate=0.001,
                   maxiter=500,
                   tol=0.1,
                   power=2,
                   update_interval=100,
                   output_save_path='selftrain_output.txt',
                   model_save_path="finetuned_model.pt"):

        # Optimizer
        if self.model == 'bert' or self.model == 'rnn':
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.classifier.parameters(), lr=learning_rate, momentum=0.9)

        # Get predictions and scores across classes
        q, y_preds = self.predict(train_loader)
        y_preds = np.asarray(y_preds)
        y_preds_last = np.copy(y_preds)

        # Pre loop variables
        output_save_path = "{}_{}_selftrain_output.txt".format(self.model, self.sup_source)
        model_save_path = "{}_model.pt".format(self.model)
        selftrain_file = open(output_save_path, 'w')
        print("LR: {}, MaxIter: {}, tol: {}, update: {}, batch: {}".format(learning_rate, maxiter, tol, update_interval, self.batch_size), file=selftrain_file)
        t0 = time()
        index = 0
        x_length = x.shape[0] if self.model in ('rnn', 'cnn') else len(y)
        index_array = np.arange(x_length)
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
                    print('f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro),file=selftrain_file)

                # Check stop criterion
                delta_label = np.sum(y_preds != y_preds_last).astype(float) / y_preds.shape[0]
                y_preds_last = np.copy(y_preds)
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label * 100, 3)))
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label * 100, 3)), file=selftrain_file)
                if ite > 0 and delta_label < tol / 100:
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label * 100, 3), tol))
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label * 100, 3), tol), file=selftrain_file)
                    print('Reached tolerance threshold. Stopping training.')
                    print('Reached tolerance threshold. Stopping training.', file=selftrain_file)
                    print('Saving...')
                    # torch.save(self.classifier, model_save_path)
                    break

            # Train on a singular batch
            idx = index_array[index * self.batch_size:min((index + 1) * self.batch_size, x_length)]
            
            if len(idx):
                if self.model == "bert":
                    temp_x = {"input_ids": x["input_ids"][idx[0]:idx[-1]], "attention_mask": x["attention_mask"][idx[0]:idx[-1]]}
                    temp_y = p[idx[0]:idx[-1]]
                    self.train_on_batch(x=temp_x, y=temp_y, batch_size=self.batch_size)
                else:
                    self.train_on_batch(x=x[idx[0]:idx[-1]], y=p[idx[0]:idx[-1]], batch_size=self.batch_size)

            index = index + 1 if (index + 1) * self.batch_size <= x_length else 0

        # Close output file
        print('Self training time: {:.2f}s'.format(time() - t0))
        print('Self training time: {:.2f}s'.format(time() - t0),file=selftrain_file)
        print('Saving...')
        torch.save(self.classifier, model_save_path)
        selftrain_file.close()

    def train_on_batch(self, x, y, batch_size):
        batch_data = DataWrapper(x, y) if self.model in ('rnn', 'cnn') else BertDataWrapper(x, y)
        batch_train_loader = DataLoader(dataset=batch_data,
                                        batch_size=batch_size,
                                        shuffle=False)
        train_loss = 0.
        train_correct = 0
        for i, (document, label) in enumerate(batch_train_loader):
            if self.model == 'rnn':
                self.classifier._init_hidden_state(self.batch_size)
                if self.is_cuda:
                    document = document.cuda()
                    label = label.cuda()
                document = Variable(document)
                feature = self.classifier(document)
            elif self.model == 'cnn':
                if self.is_cuda:
                    document = document.cuda()
                    label = label.cuda()
                document = Variable(document)
                feature = self.classifier(document)
            elif self.model == 'bert':
                input_id, mask = document
                if self.is_cuda:
                    input_id = input_id.cuda()
                    mask = mask.cuda()
                    label = label.cuda()
                feature = self.classifier(input_id, mask)

            # Get categorical target
            train_correct += binary_accuracy(feature.cpu().detach(),
                                             label.cpu().detach())

            # Compute Loss
            loss = self.criterion(feature, label)

            # Clear gradient in optimizer
            self.optimizer.zero_grad()
            loss.backward()

            # Do one step of gradient descent
            self.optimizer.step()
            train_loss += loss.item()
        
    
    def target_distribution(self, q, power=2):
        # square each class, divide by total of class
        weight = q**power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        for i in self.sup_list:
            p[i] = 0
            p[i][self.sup_list[i]] = 1
        return p