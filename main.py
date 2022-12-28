from os import lseek, truncate
from model import WSTC
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from argparse import RawTextHelpFormatter
from utils.datahelper import DataWrapper, BertDataWrapper
from architectures.transformer import *
from utils.load_data import load_dataset, train_word2vec
from utils.gen import *
from utils.bert_utils import *
from architectures.bag_of_words import bag_of_words


def main():
    ### Arg Parser Settings ###
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument("--data", default="generate", choices=["generate", "load"])
    parser.add_argument("--model", default="cnn", choices=["rnn", "bert", "cnn"])
    parser.add_argument('--sup_source', default='keywords', choices=['labels', 'keywords', 'docs'])
    parser.add_argument('--pretrain', default='False', action="store_true")
    parser.add_argument('--selftrain', default='False', action="store_true")
    parser.add_argument('--evaluate', default='False', action="store_true")
    parser.add_argument('--with_statistics', default='False', action="store_true")
    parser.add_argument('--load_model', default='False', action="store_true")
    parser.add_argument('--save_docs', default='False', action="store_true")
    parser.add_argument('--aware', default='False', action="store_true")
    parser.add_argument("--aware_type", default="w2v", choices=["w2v", "avg"])

    args = parser.parse_args()
    print(args)

    ### Hyperparamter Settings ###
    if args.model == 'rnn':
        doc_len = 10
        sent_len = 45
        sequence_length = [doc_len, sent_len]
        batch_size = 256
        pretrain_epochs = 20
        lr = 0.001
        self_lr = 0.0001
        update_interval = 50
        maxiter = 5000
    elif args.model == 'cnn':
        sequence_length = 100
        batch_size = 256
        pretrain_epochs = 20
        lr = 0.1
        self_lr = 0.001
        update_interval = 50
        maxiter = 5000
    elif args.model == 'bert':
        batch_size = 16
        pretrain_epochs = 3
        lr = 0.00005
        self_lr = 0.00001
        update_interval = 500
        maxiter = 3000
    sup_idx = None

    ### Data Section ###
    embedding_mat = None
    if args.data == "generate":
        if args.model in ("rnn", "cnn"):

            if args.aware == True:
                print('here')
                # seed_docs, seed_label = \
                #     load_data_bert(sup_source=args.sup_source, with_evaluation=args.with_statistics, gen_seed_docs=args.data, model_type=args.model)


                if args.aware_type == 'w2v':
                    # Train Word2Vec embedding matrix
                    embedding_mat, vocab, vocab_inv = trainW2V_BERT_Tokenizer()
                elif args.aware_type == 'avg':
                    # Get BERT average embeddings
                    embedding_mat, vocab, vocab_inv = createBERTEmbeddingVocab()

                # Convert BERT docs to vocab doc
                seed_docs_numpy = np.load("data/seed_docs_bert_5000.npy", allow_pickle=True).item()
                seed_label = np.load("data/seed_labels_bert_5000.npy")
                seed_docs_numpy = bert_encodings_to_vocab_encodings(seed_docs_numpy, vocab).astype(np.int32)
                seed_docs = torch.from_numpy(seed_docs_numpy)
                # print(seed_docs.shape)
                print(seed_label.shape)
                # print().shape

                # x = np.load('data/bert_data.npy')
                # y = np.load('data/real_label_bert.npy')
                x = np.load("data/real_docs_bert_5000.npy", allow_pickle=True).item()
                y = np.load("data/real_labels_bert_5000.npy")
                x = bert_encodings_to_vocab_encodings(x, vocab).astype(np.int32)
                # print(y)
                # bag_of_words(x, y)
                # print().shape



                # x = bert_encodings_to_vocab_encodings(x, vocab)
                # print(x.shape)
                # print(y.shape)
                # if args.model == "bert":
                #     seed_attention_mask = np.zeros((5000 * 4, 72))
                #     seed_attention_mask[:, :39] = 1
                #     seed_docs = {"input_ids": seed_docs, "attention_mask": seed_attention_mask}
                #     x_attention_mask = np.zeros((5000 * 4, 72))
                #     x_attention_mask[:, :39] = 1
                #     x = {"input_ids": x, "attention_mask": x_attention_mask}
                # print(seed_docs.shape)
                # print().shape

            else:
                x, y, word_counts, vocabulary, vocabulary_inv_list, len_avg, len_std, word_sup_list, sup_idx, perm = \
                    load_dataset(args.model, sup_source=args.sup_source, with_evaluation=args.with_statistics, truncate_len=sequence_length)

                # Truncate data
                if args.model == 'rnn':
                    x = x[:, :doc_len, :sent_len]
                elif args.model == 'cnn':
                    x = x[:, :100]
                
                # Create embedding matrix
                vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
                embedding_weights = train_word2vec(x, vocabulary_inv, "agnews")
                embedding_mat = np.array([np.array(embedding_weights[word]) for word in vocabulary_inv])

                seed_docs, seed_label = generate_pseudocs(args.model, embedding_mat, word_sup_list, vocabulary_inv_list, 
                                                        word_counts, sequence_length, vocabulary, len_avg, len_std)

                # Delete later
                # decode_docs(seed_docs, vocabulary_inv)
                # print().shape
            if args.sup_source == 'docs':
                if args.model == 'cnn':
                    num_real_doc = len(sup_idx.flatten()) * 10
                elif args.model == 'rnn':
                    num_real_doc = len(sup_idx.flatten())
                real_seed_docs, real_seed_label = augment(x, sup_idx, num_real_doc)
                seed_docs = np.concatenate((seed_docs, real_seed_docs), axis=0)
                seed_label = np.concatenate((seed_label, real_seed_label), axis=0)

        elif args.model == "bert":
            x, y, seed_docs, seed_label = \
                load_data_bert(sup_source=args.sup_source, with_evaluation=args.with_statistics, gen_seed_docs=args.data)

        if args.save_docs == True:
            print("Saving docs...")
            np.save("data/seed_docs_{}.npy".format(args.model), seed_docs)
            np.save("data/seed_label_{}.npy".format(args.model), seed_label)
            np.save('data/real_docs_{}.npy'.format(args.model), x)
            np.save('data/real_label_{}.npy'.format(args.model), y)
            # if args.model in ("rnn", "cnn"):
            #     np.save('data/embedding_matrix.npy', embedding_mat)

    elif args.data == "load":
        if args.pretrain == True:
            if args.model in ("rnn", "cnn"):
                seed_docs_numpy = np.load('data/seed_docs_{}.npy'.format(args.model))
                seed_label = np.load('data/seed_label_{}.npy'.format(args.model))
                seed_docs = torch.from_numpy(seed_docs_numpy)
                print(seed_docs.shape)
                # seed_docs = seed_docs[:, :72]
                print(seed_docs.shape)
                embedding_mat = np.load('data/embedding_matrix.npy')
                # print().shape
            elif args.model == "bert":
                # seed_docs = np.load('data/seed_docs_{}_5000.npy'.format(args.model), allow_pickle=True).item()
                # seed_label = np.load('data/seed_labels_{}_5000.npy'.format(args.model))
                #### delete ####
                seed_docs_numpy = np.load('data/seed_docs_{}.npy'.format("cnn"))
                seed_docs = torch.from_numpy(seed_docs_numpy)
                seed_label = np.load('data/seed_label_{}.npy'.format("cnn"))
                vocabulary_inv_list = np.load('vocabulary_inv_list.npy')
                seed_docs = tokenizeText(seed_docs_numpy, vocabulary_inv_list)
                #### delete ####
        if args.selftrain == True or args.evaluate == True:
            if args.model in ("rnn", "cnn"):
                x = np.load('data/real_docs_{}.npy'.format(args.model))
                y = np.load('data/real_label_{}.npy'.format(args.model))
                embedding_mat = np.load('data/embedding_matrix.npy')
            elif args.model == "bert":
                # x = np.load('data/real_docs_{}_5000.npy'.format(args.model))
                # x = np.load('data/real_docs_{}_5000.npy'.format(args.model), allow_pickle=True).item()
                # y = np.load('data/real_labels_{}_5000.npy'.format(args.model))
                #### delete ####
                x = np.load('data/real_docs_{}.npy'.format("cnn"))
                y = np.load('data/real_label_{}.npy'.format("cnn"))
                vocabulary_inv_list = np.load('vocabulary_inv_list.npy')
                x = tokenizeText(x, vocabulary_inv_list)
                #### delete ####

    ### Model Instantiation ###
    classifier = torch.load("{}_model.pt".format(args.model)) if args.load_model == True else None
    wstc = WSTC(model=args.model,
                batch_size=batch_size,
                embedding_matrix=embedding_mat,
                learning_rate=lr,
                classifier=classifier,
                sup_source=args.sup_source)

    if args.pretrain == True:
        seed_label = torch.from_numpy(seed_label)
        seed_label = seed_label.type(torch.FloatTensor)
        train_data = DataWrapper(seed_docs, seed_label) if args.model in ('rnn', 'cnn') else BertDataWrapper(seed_docs, seed_label)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        wstc.pretrain(train_loader, pretrain_epochs, sup_idx=sup_idx)

    if args.selftrain == True:

        # x = np.load('data/real_docs_{}.npy'.format(args.model))
        # y = np.load('data/real_label_{}.npy'.format(args.model))
        # Randomize because can't shuffle batches

        p = np.random.permutation(len(y))
        if args.model == "bert":
            x = {"input_ids": x["input_ids"][p], "attention_mask": x["attention_mask"][p]}
        else:
            x = x[p]
        y = y[p]
        self_train_data = DataWrapper(x, y) if args.model in ('rnn', 'cnn') else BertDataWrapper(x, y)
        self_train_loader = DataLoader(dataset=self_train_data, batch_size=batch_size, shuffle=False)
        print('here')
        wstc.self_train(self_train_loader, x, y, learning_rate=self_lr, maxiter=maxiter, update_interval=update_interval)


    if args.evaluate == True:
        # ###### Delete later
        # x = seed_docs
        # y = torch.argmax(seed_label.cpu(), dim=1)
        ###### Delete later
        test_data = DataWrapper(x, y) if args.model in ('rnn', 'cnn') else BertDataWrapper(x, y)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        wstc.evaluate_dataset(test_loader)

if __name__ == "__main__":
    main()