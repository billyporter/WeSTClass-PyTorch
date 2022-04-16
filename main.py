from model import WSTC
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from layers import DataWrapper
from HAN.hier_att_model import HierAttNet
from HAN.utils import get_evaluation
import argparse
from argparse import RawTextHelpFormatter
from layers import DataWrapper
from transformer import *


def main():

    # Add parser arguments
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    ### Basic Settings ###
    parser.add_argument("-m",
                        "--method",
                        choices=["p", "s", "e", "a", "t", "b"],
                        help="p: pretrain\ns: self-train\ne: eval\na: pretrain with test\nt: train on real documents")

    ### Training Settings ###
    # mini-batch size for both pre-training and self-training: 256 (default)
    parser.add_argument('--batch_size', default=256, type=int)
    # maximum self-training iterations: 5000 (default)
    parser.add_argument('--maxiter', default=5e3, type=int)
    # pre-training epochs: None (default)
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    # self-training update interval: None (default)
    parser.add_argument('--update_interval', default=50, type=int)
    # data source (truncated or full, default full)
    parser.add_argument(
        '--data',
        choices=["full", "trunc"],
        default="trunc",
        help="full: full 120,000 documents\ntrunc: 2,000 documents")
    args = parser.parse_args()


    ### Necessary Setup ###
    # Data paths
    docs_path = "data/real_docs_full.npy"
    labels_path = "data/real_labels_full.npy"

    if args.data == "trunc":
        docs_path = "data/real_docs_trunc.npy"
        labels_path = "data/real_labels_trunc.npy"

    # Constants from psuedo-doc generation
    xshape = (12000, 10, 45)
    n_classes = 4
    argsmodel = 'rnn'
    vocab_sz = 67765
    word_embedding_dim = 100
    
    # Load embedding matrix
    embedding_mat = np.load('data/embedding_mat.npy')

    ### Pretrain with Test ###
    if args.method == 'a':
        
        # Load Seed Documents
        seed_docs_numpy = np.load('data/seed_docs.npy')
        seed_label = np.load('data/seed_label.npy')

        # Process
        seed_docs = torch.from_numpy(seed_docs_numpy)
        seed_label = torch.from_numpy(seed_label)
        seed_label = seed_label.type(torch.FloatTensor)

        # Convert to batches of tensors
        train_data = DataWrapper(seed_docs, seed_label)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True)


        # Load Real Documents
        x = np.load(docs_path)
        y = np.load(labels_path)
        
        real_train_data = DataWrapper(x, y)
        real_train_loader = DataLoader(dataset=real_train_data,
                                       batch_size=args.batch_size,
                                       shuffle=False)
        
        
        # Create Model
        wstc = WSTC(input_shape=xshape,
                    n_classes=n_classes,
                    model=argsmodel,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim)

        # Call method
        wstc.pretrain_with_test(train_loader, args.pretrain_epochs,
                                real_train_loader)

    ### Pretrain ###
    if args.method == 'p':
        
        # Load Seed Documents 
        seed_docs_numpy = np.load('data/seed_docs.npy')
        seed_label = np.load('data/seed_label.npy')

        # Process
        seed_docs = torch.from_numpy(seed_docs_numpy)
        seed_label = torch.from_numpy(seed_label)
        seed_label = seed_label.type(torch.FloatTensor)

        # Convert to batches of tensors
        train_data = DataWrapper(seed_docs, seed_label)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True)

        # Create Model
        wstc = WSTC(input_shape=xshape,
                    n_classes=n_classes,
                    model=argsmodel,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim)

        wstc.pretrain(train_loader, args.pretrain_epochs)

    ### Self train ###
    if args.method == 's':

        # Load saved model
        model = torch.load('model.pt')

        # Load real documents
        
        x = np.load(docs_path)
        y = np.load(labels_path)
        
        x = x[:, :10, :45]
        
        # Convert to batches of tensors
        self_train_data = DataWrapper(x, y)
        self_train_loader = DataLoader(dataset=self_train_data,
                                       batch_size=args.batch_size,
                                       shuffle=False)

        # Create Model
        wstc = WSTC(input_shape=xshape,
                    classifier=model,
                    n_classes=n_classes,
                    model=argsmodel,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim)

        # Call method
        wstc.self_train(self_train_loader,
                        x,
                        y,
                        maxiter=args.maxiter,
                        update_interval=args.update_interval)

    ### Evaluuate ###
    if args.method == 'e':

        # Load saved model
        model = torch.load('model.pt')

        # Load real documents
        x = np.load(docs_path)
        y = np.load(labels_path)

        # Convert to batches of tensors
        test_data = DataWrapper(x, y)
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)

        # Create object
        wstc = WSTC(input_shape=xshape,
                    classifier=model,
                    n_classes=n_classes,
                    model=argsmodel,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim)

        # Cal method
        wstc.evaluate_dataset(test_loader)
        
    ### Train on real documents ###
    if args.method == 't':
        
        # Load real documents
        docs_numpy = np.load("data/real_docs_full.npy")
        labels = np.load("data/real_labels_full.npy")

        # Change to one hot encoding
        new_labels = np.full((labels.size, labels.max() + 1), 0.0)
        new_labels[np.arange(labels.size), labels] = 1
        labels = new_labels

        # Select 500 random from each class
        perm_seed = np.random.choice(np.arange(len(docs_numpy) / 4),
                                     500,
                                     replace=False)
        first_seed = perm_seed
        sec_seed = first_seed * 2
        third_seed = first_seed * 3
        fourth_seed = first_seed * 4

        all_perm_seeds = np.concatenate((first_seed, sec_seed, third_seed, fourth_seed),axis=0).astype(int)

        docs_numpy = docs_numpy[all_perm_seeds]
        labels = labels[all_perm_seeds]

        # Process
        docs = torch.from_numpy(docs_numpy)
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.FloatTensor)

        # Convert to batches of tensors
        train_data = DataWrapper(docs, labels)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True)

        # Create model
        wstc = WSTC(input_shape=xshape,
                    n_classes=n_classes,
                    model=argsmodel,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim)

        # Call method
        wstc.pretrain(train_loader, args.pretrain_epochs)


    if args.method == 'b':
        from transformers import BertTokenizer
        seed_docs_text = np.load('data/seed_docs_text.npy')
        seed_label = np.load('data/seed_label.npy')

        # Process
        seed_label = torch.from_numpy(seed_label)
        seed_label = seed_label.type(torch.FloatTensor)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        bert_input = tokenize_bert(seed_docs_text, tokenizer)

        train_data = DataWrapper(bert_input, seed_label)
        train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

        bert = BertClassifier()

        train_bert(bert, train_loader)



        


if __name__ == "__main__":
    main()
