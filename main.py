from os import truncate
from model import WSTC
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from layers import DataWrapper
import argparse
from argparse import RawTextHelpFormatter
from layers import DataWrapper
from transformer import *
from load_data import load_dataset
from gen import *
from bert_utils import *


def main():

    # Add parser arguments
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    ### Basic Settings ###
    parser.add_argument("--data", default="generate", choices=["generate", "load"])
    parser.add_argument("--model", default="rnn", choices=["rnn", "bert"])
    parser.add_argument("--method", default="pretrain", choices=["pretrain", "selftrain", "both", "neither"])
    parser.add_argument("--evaluate", default="True", choices=["True", "False"])
    parser.add_argument("--with_statistics", default="True", choices=["True", "False"])
    parser.add_argument("--trained_weights", default=None)

    ### Training Settings ###
    # mini-batch size for both pre-training and self-training: 256 (default)
    parser.add_argument('--batch_size', default=256, type=int)
    # maximum self-training iterations: 5000 (default)
    parser.add_argument('--maxiter', default=9e3, type=int)
    # pre-training epochs: None (default)
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    # self-training update interval: None (default)
    parser.add_argument('--update_interval', default=250, type=int)


    args = parser.parse_args()
    print(args)
    print('a')


    doc_len = 10
    sent_len = 45
    truncate_len = [doc_len, sent_len]

    if args.method in ("selftrain", "both") or args.data == "generate":
        # x, y, word_counts, vocabulary, vocabulary_inv_list, len_avg, len_std, word_sup_list, perm = \
        #     load_dataset(with_evaluation=args.with_statistics, truncate_len=truncate_len)
        # print('word_sup_list: ', word_sup_list)

        x, y, seed_docs, seed_label = \
            load_data_bert(with_evaluation=args.with_statistics, truncate_len=truncate_len)
        print().shape

    ### Load Data
    if args.data == "generate":

        x = x[:, :doc_len, :sent_len]
        sequence_length = [doc_len, sent_len]

        print("\n### Input preparation ###")
        embedding_mat = np.load('data/embedding_mat.npy')

        print("\n### Phase 1: vMF distribution fitting & pseudo document generation ###")
        seed_docs, seed_label = generate_pseudocs(embedding_mat, word_sup_list, vocabulary_inv_list, 
                                                word_counts, sequence_length, vocabulary, len_avg, len_std)

        perm_seed = np.random.permutation(len(seed_label))
        seed_docs = seed_docs[perm_seed]
        seed_label = seed_label[perm_seed]


        if args.model == "bert":
            seed_docs = tokenizeText(x, vocabulary_inv_list)
            # seed_docs = tokenizeText(seed_docs_numpy, vocabulary_inv_list)


    elif args.data == "load":
        seed_docs_numpy = np.load('data/seed_docs.npy')

        # Process
        seed_label = np.load('data/seed_label.npy')
        seed_docs = torch.from_numpy(seed_docs_numpy)

        if args.model == "bert":
            vocabulary_inv_list = np.load('vocabulary_inv_list.npy')
            seed_docs = tokenizeText(seed_docs_numpy, vocabulary_inv_list)


    seed_label = torch.from_numpy(seed_label)
    seed_label = seed_label.type(torch.FloatTensor)


    
    # # Constants from psuedo-doc generation
    xshape = (12000, 10, 45)
    vocab_sz = 67765
    word_embedding_dim = 100


    # # Load embedding matrix
    embedding_mat = np.load('data/embedding_mat.npy')

    # lr = 0.01 if args.model == 'rnn' else 0.0001
    lr = 0.01 if args.model == 'rnn' else 0.0001
    wstc = WSTC(input_shape=xshape,
                model=args.model,
                batch_size=args.batch_size,
                vocab_sz=vocab_sz,
                embedding_matrix=embedding_mat,
                word_embedding_dim=word_embedding_dim,
                learning_rate=lr)

    if args.method == "pretrain" or args.method == "both":
        train_data = DataWrapper(seed_docs, seed_label)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
        wstc.pretrain(train_loader, args.pretrain_epochs)

    if args.method == "selftrain" or args.method == "both":

        # REMOVE SOON
        docs_path = "data/real_docs_full.npy"
        labels_path = "data/real_labels_full.npy"

        x = np.load(docs_path)
        y = np.load(labels_path)
        
        x = x[:, :10, :45]
        # y = y[:200]

        p = np.random.permutation(len(x))
        x = x[p]
        y = y[p]

        # print(x.shape)

        ### DELETE
        vocabulary_inv_list = np.load('vocabulary_inv_list.npy')
        x = tokenizeText(x, vocabulary_inv_list)
        # x = np.asarray(x)
        # print(len(x))
        
        
        # Convert to batches of tensors
        self_train_data = DataWrapper(x, y)
        self_train_loader = DataLoader(dataset=self_train_data,
                                       batch_size=args.batch_size,
                                       shuffle=False)
        model = torch.load('model.pt')

        wstc = WSTC(input_shape=xshape,
                    classifier=model,
                    model=args.model,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim,
                    learning_rate=lr)

        wstc.self_train(self_train_loader,
                        x,
                        y,
                        maxiter=args.maxiter,
                        update_interval=args.update_interval)


    if args.evaluate:

        docs_path = "data/real_docs_full.npy"
        labels_path = "data/real_labels_full.npy"

        # Load saved model
        model = torch.load('finetuned_model.pt')

        # Load real documents
        x = np.load(docs_path)
        y = np.load(labels_path)
        x = x[:, :10, :45]

        if args.model == 'bert':
            vocabulary_inv_list = np.load('vocabulary_inv_list.npy')
            x = tokenizeText(x, vocabulary_inv_list)

        # x = x[:, :10, :45]

        # Convert to batches of tensors
        test_data = DataWrapper(x, y)
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)

        # Create object
        wstc = WSTC(input_shape=xshape,
                    classifier=model,
                    model=args.model,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim)

        # Cal method
        wstc.evaluate_dataset(test_loader)

if __name__ == "__main__":
    main()