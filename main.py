from model import WSTC
import numpy as np
import torch
from torch.utils.data import DataLoader

class DataWrapper():
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def main():
    print(torch. __version__)

    # Constants
    xshape = (12000, 10, 45)
    n_classes = 4
    argsmodel = 'rnn'
    vocab_sz = 67765
    word_embedding_dim = 100

    # Load data
    seed_docs_numpy = np.load('seed_docs.npy')
    seed_label = np.load('seed_label.npy')
    embedding_mat = np.load('embedding_mat.npy')

    # Process
    seed_docs = torch.from_numpy(seed_docs_numpy)
    seed_label = torch.from_numpy(seed_label)
    seed_docs = seed_docs.long()
    seed_label = seed_label.type(torch.FloatTensor)
    print('here')


    # Process data
    train_data = DataWrapper(seed_docs, seed_label)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)   

    # Call model
    # wstc = WSTC(input_shape=xshape, n_classes=n_classes, model=argsmodel,
    #             vocab_sz=vocab_sz, embedding_matrix=embedding_mat, word_embedding_dim=word_embedding_dim)

    # wstc.pretrain(train_loader)

    model = torch.load('model')

    y = np.load('new_y.npy')
    x = np.load('new_x.npy')
    train_data = DataWrapper(x, y)
    self_train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)

    wstc = WSTC(input_shape=xshape, classifier=model, n_classes=n_classes, model=argsmodel,
                vocab_sz=vocab_sz, embedding_matrix=embedding_mat, word_embedding_dim=word_embedding_dim)

    wstc.self_train(self_train_loader, x, y)





if __name__ == "__main__":
    main()