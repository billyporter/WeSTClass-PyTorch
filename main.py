from model import WSTC
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import DataWrapper
import argparse
from argparse import RawTextHelpFormatter


class DataWrapper():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def main():

    # Add parser arguments
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    ### Basic Settings ###
    parser.add_argument("-m",
                        "--method",
                        choices=["p", "s", "e"],
                        help="p: pretrain\ns: self-train\ne: eval")

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
    parser.add_argument('--data', 
                        choices=["full", "trunc"],
                        default="trunc",
                        help="full: full 120,000 documents\ntrunc: 2,000 documents")
    args = parser.parse_args()

    ### Data
    docs_path           = "data/real_docs_full.npy"
    labels_path         = "data/real_labels_full.npy"

    if args.data == "trunc":
        docs_path       = "data/real_docs_trunc.npy"
        labels_path     = "data/real_labels_trunc.npy"


    ### Constants ###
    xshape = (12000, 10, 45)
    n_classes = 4
    argsmodel = 'rnn'
    vocab_sz = 67765
    word_embedding_dim = 100
    embedding_mat = np.load('data/embedding_mat.npy')

    ### Pretrain ###
    if args.method == 'p':
        # Load data
        seed_docs_numpy = np.load('data/seed_docs.npy')
        seed_label = np.load('data/seed_label.npy')

        # Process
        seed_docs = torch.from_numpy(seed_docs_numpy)
        seed_label = torch.from_numpy(seed_label)
        seed_label = seed_label.type(torch.FloatTensor)

        # Process data
        train_data = DataWrapper(seed_docs, seed_label)
        train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)

        # Call model
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

        # # Load data
        x = np.load(docs_path)
        y = np.load(labels_path)
        # print(x.shape)
        train_data = DataWrapper(x, y)
        self_train_loader = DataLoader(dataset=train_data,
                                       batch_size=args.batch_size,
                                       shuffle=False)

        # Call self training
        wstc = WSTC(input_shape=xshape,
                    classifier=model,
                    n_classes=n_classes,
                    model=argsmodel,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim)

        wstc.self_train(self_train_loader,
                        x,
                        y,
                        maxiter=args.maxiter,
                        update_interval=args.update_interval)

    if args.method == 'e':

        # Load saved model
        model = torch.load('model.pt')

        # Load data
        # model = None
        x = np.load(docs_path)
        y = np.load(labels_path)

        test_data = DataWrapper(x, y)
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)

        wstc = WSTC(input_shape=xshape,
                    classifier=model,
                    n_classes=n_classes,
                    model=argsmodel,
                    batch_size=args.batch_size,
                    vocab_sz=vocab_sz,
                    embedding_matrix=embedding_mat,
                    word_embedding_dim=word_embedding_dim)

        wstc.evaluate_dataset(test_loader)


if __name__ == "__main__":
    main()
