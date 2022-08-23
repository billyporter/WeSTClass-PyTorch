from base64 import decode, encode
from distutils.command.build import build
from tkinter.filedialog import test
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import re
import torch
from statistics import mean
import random
from os.path import join
from time import time
from utils.gen_bert import *
from utils.load_data import read_file, load_keywords
from utils.datahelper import DataWrapper, BertDataWrapper
import sys

# New
import itertools
from collections import Counter
from gensim.models import word2vec

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?_\"\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def preprocess_doc(data):
    data = [s.strip() for s in data]
    data = [clean_str(s) for s in data]
    return data

def get_statistics(data):
    
    data = [s.split(" ") for s in data]
    tmp_list = [len(doc) for doc in data]

    doc_lens_list = [len(row) for row in data]
    max_len = max(tmp_list)
    avg_len = np.average(tmp_list)
    std_len = np.std(tmp_list)

    print("----------------------------------------}")
    print("Max document length: {}".format(max_len))
    print("Average document length: {}".format(avg_len))
    print("St. Dev document length: {}".format(std_len))

    real_len = min(int(avg_len + 3 * std_len), max_len)
    print("Defined document length: {}".format(real_len))
    print('Fraction of truncated sentences: {}'.format(sum(tmp > real_len for tmp in doc_lens_list)/len(doc_lens_list)))

    pseudo_len = int(np.floor(avg_len))
    print("Defined psuedo length: {}".format(pseudo_len))
    print("----------------------------------------}")


    return real_len, pseudo_len, max_len

def extract_keywords(data_path, num_keywords, data, perm):
    sup_data = []
    sup_idx = []
    sup_label = []
    file_name = 'doc_id.txt'
    infile = open(join(data_path, file_name), mode='r', encoding='utf-8')
    text = infile.readlines()

    for i, line in enumerate(text):
        line = line.split('\n')[0]
        class_id, doc_ids = line.split(':')
        assert int(class_id) == i
        seed_idx = doc_ids.split(',')
        seed_idx = [int(idx) for idx in seed_idx]
        sup_idx.append(seed_idx)
        for idx in seed_idx:
            print(data[idx])
            sup_data.append(" ".join(data[idx]))
            sup_label.append(i)

    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk

    tfidf = TfidfVectorizer(norm='l2', sublinear_tf=True, max_df=0.2, stop_words='english')
    sup_x = tfidf.fit_transform(sup_data)
    sup_x = np.asarray(sup_x.todense())

    vocab_dict = tfidf.vocabulary_
    vocab_inv_dict = {v: k for k, v in vocab_dict.items()}

    print("\n### Supervision type: Labeled documents ###")
    print("Extracted keywords for each class: ")
    keywords = []
    cnt = 0
    for i in range(len(sup_idx)):
        class_vec = np.average(sup_x[cnt:cnt+len(sup_idx[i])], axis=0)
        cnt += len(sup_idx[i])
        sort_idx = np.argsort(class_vec)[::-1]
        keyword = []
        j = 0
        k = 0
        while j < num_keywords:
            w = vocab_inv_dict[sort_idx[k]]
            keyword.append(vocab_inv_dict[sort_idx[k]])
            j += 1
            k += 1
        print("Class {}:".format(i))
        print(keyword)
        keywords.append(keyword)

    new_sup_idx = []
    m = {v: k for k, v in enumerate(perm)}
    for seed_idx in sup_idx:
        new_seed_idx = []
        for ele in seed_idx:
            new_seed_idx.append(m[ele])
        new_sup_idx.append(new_seed_idx)
    new_sup_idx = np.asarray(new_sup_idx)

    return keywords, new_sup_idx

def reduce_data(x, y, num_examples, num_classes):
    print(len(x))
    class_perms_list = []
    old_class_sz = len(x) // num_classes
    for i in range(1, num_classes + 1):
        indices = np.arange((i - 1) * old_class_sz, i * old_class_sz)
        perm = np.random.permutation(indices)
        class_perms_list.extend(perm[:num_examples])
    p = np.asarray(class_perms_list)
    x = np.asarray(x)[p]
    y = np.asarray(y)[p]
    print(x.shape)
    return list(x), list(y)

def reduce_embeddings(embed_encode_dict, encode_count_dict, n):
    embed_list = embed_encode_dict.keys()

    # 1) Construct list of embeddings counts
    embeddings_counts = [encode_count_dict[embed_encode_dict[embed]] 
        for embed in embed_list]

    # 2) Get how indices should be sorted
    sorted_indices = np.argsort(embeddings_counts)

    # 3) Get corresponding embeddings and encodings
    embedding_mat = np.array(list(embed_encode_dict.keys()))[sorted_indices]
    encode_list = np.array(list(embed_encode_dict.values()))[sorted_indices]
    # embedding_mat = embedding_mat[:n]
    # encode_list = encode_list[:n]

    return embedding_mat, encode_list

def get_keyword_emebds(keywords, encode_count_dict, encode_embed_dict):
    pass

def load_data_bert(dataset_name="agnews", sup_source="keywords", with_evaluation=True, gen_seed_docs="generate", batch_size=16, model_type="bert"):
    data_path = 'data/' + dataset_name
    data, y_full = read_file(data_path, with_evaluation)

    # Clean data
    data_full = preprocess_doc(data)

    # Get subset of data for embeddings
    data, y = reduce_data(data_full, y_full, 2500, 4)
    print(len(data))

    # Get data statistics
    real_len, pseudo_len, embedding_len = get_statistics(data)
    print(real_len, pseudo_len)

    # Add keywords to end sentence
    if sup_source in ("labels", "keywords"):
        keywords = load_keywords(data_path, sup_source)
    elif sup_source == "docs":
        sz = len(data)
        np.random.seed(1234)
        perm = np.random.permutation(sz)
        data_copy = [s.split(" ") for s in data]
        keywords, sup_idx = extract_keywords(data_path, 10, data_copy, perm)
    keywords_sents = [" ".join(sent) for sent in keywords]
    print(keywords)
    print('Done preprocessing...')

    # Tokenize data
    print("Converting text to tokens...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_data = tokenizer(data, padding='max_length', max_length=real_len, truncation=True,)
    # encoded_data_full = tokenizer(data_full, padding='max_length', max_length=real_len, truncation=True,)
    encoded_keywords = tokenizer(keywords_sents, padding='max_length', max_length=real_len, truncation=True)
    print('Done tokenizing...')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )
    model.eval()

    # Prepare data for model
    tokens_tensor = torch.tensor(encoded_data["input_ids"])
    segments_tensor = torch.tensor(encoded_data["token_type_ids"])
    bert_data = DataWrapper(tokens_tensor, segments_tensor)
    bert_loader = DataLoader(dataset=bert_data, batch_size=batch_size, shuffle=False)

    # Keywords model prep
    tokens_tensor_keywords = torch.tensor(encoded_keywords["input_ids"])
    segments_tensor_keywords = torch.tensor(encoded_keywords["token_type_ids"])

    # Construct embedding: id dictionary
    embed_encode_dict = {}
    encode_count_dict = {}
    keyword_embeds = []
    keyword_encodes = []

    # Constants
    PAD_encoding = 0
    CLS_encoding = 101
    SEP_encoding = 102

    # Get keywords encodes
    all_keyword_encodes = {}
    keyword_class_dict = {}
    for i, encoding_class in enumerate(encoded_keywords["input_ids"]):
        keyword_embeds.append([])
        temp_encodes = []
        for encoding in encoding_class:
            if encoding == CLS_encoding:
                continue
            if encoding == SEP_encoding:
                continue
            if encoding == PAD_encoding:
                break
            all_keyword_encodes[encoding] = []
            keyword_class_dict[encoding] = i
            temp_encodes.append(encoding)
        keyword_encodes.append(temp_encodes)

    print("Beginning Embedding Generation...")
    # Embedding loop
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        # Loop through data
        for i, (tokens_tensor, segments_tensor) in enumerate(tqdm(bert_loader)):
            if torch.cuda.is_available():
                tokens_tensor = tokens_tensor.cuda()
                segments_tensor = segments_tensor.cuda()

            outputs = model(tokens_tensor, segments_tensor)

            # Get second to last hidden state
            hidden_states = outputs[2]
            last_hidden_state = hidden_states[-2]
            
            # Sum Last four layers
            # shape : [16, 71, 768])
            # batch x tokens x dimensions
            # last_hidden_state = torch.stack(hidden_states[-4:]).sum(0)

            # Add embeddings to dictionary
            token_embeddings = torch.squeeze(last_hidden_state, dim=1)
            for i, document in enumerate(tokens_tensor):
                for j, encoding in enumerate(document):
                    if encoding == CLS_encoding:
                        continue
                    if encoding == SEP_encoding:
                        continue
                    if encoding == PAD_encoding:
                        break
                    embedding = token_embeddings[i][j].cpu().numpy()
                    embedding = tuple(embedding / np.linalg.norm(embedding))
                    encoding = encoding.cpu().numpy().item()
                    embed_encode_dict[embedding] = encoding
                    encode_count_dict[encoding] = encode_count_dict.get(encoding, 0) + 1

                    # NEW #
                    if encoding in all_keyword_encodes:
                        all_keyword_encodes[encoding].append(embedding)
                    # NEW #
        
    print("Finished Embedding Generation...")
    print("encoded data size: ", sys.getsizeof(encoded_data))
    print("embed dict size: ", sys.getsizeof(embed_encode_dict))
    print("encode dict size: ", sys.getsizeof(encode_count_dict))

    # NEW #

    for encoding, embed_candidates in all_keyword_encodes.items():
        # Get class index
        class_index = keyword_class_dict[encoding]

        # Get highest embed
        # Get average embedding
        if len(embed_candidates):
            word_embeddings = torch.zeros((len(embed_candidates), 768))
            for j, embed in enumerate(embed_candidates):
                word_embeddings[j] = torch.FloatTensor(embed)
            combined_embedding = torch.mean(word_embeddings, 0)

            # Add to embeds
            keyword_embeds[class_index].append(tuple(combined_embedding.tolist()))
    
    # NEW #

    seed_docs_input_ids = []
    seed_labels = []
    if gen_seed_docs == "generate":
        # Create embedding matrix

        # embedding_mat, encode_list = reduce_embeddings(embed_encode_dict, encode_count_dict, 50000)
        embedding_mat = np.array(sorted(list(embed_encode_dict.keys())))
        encode_list = np.array(list(embed_encode_dict.values()))
        # print("embed mat len: ", len(embedding_mat))
        # print("embed mat size: ", sys.getsizeof(embedding_mat))
    

        # Get VMF Distribution
        _, centers, kappas = label_expansion_bert(embedding_mat, embed_encode_dict, keyword_embeds, keyword_encodes, tokenizer)
        print("embed mat len: ", len(embedding_mat))
        print("embed mat size: ", sys.getsizeof(embedding_mat))

        # Generate Pseudo Documents
        print("Pseudo documents generation...")
        num_doc_per_class = 500
        vocab_size = 50
        interp_weight = 0.2
        sequence_length = pseudo_len
        doc_settings = (num_doc_per_class, vocab_size, interp_weight, real_len, sequence_length)
        seed_docs_input_ids, seed_labels = \
            psuedodocs_bert(embedding_mat, encode_count_dict,
            keyword_encodes, encode_list, centers, kappas, doc_settings)

        # Conver to correct form
        seed_attention_mask = np.zeros((num_doc_per_class * len(keyword_encodes), real_len))
        seed_attention_mask[:, :sequence_length] = 1

    seed_docs = {"input_ids": seed_docs_input_ids, "attention_mask": seed_attention_mask}
    # encoded_data_full = {"input_ids": np.asarray(encoded_data_full["input_ids"]),
    #  "attention_mask": np.asarray(encoded_data_full["attention_mask"])}
    # return encoded_data_full, y_full, seed_docs, seed_labels
    # np.save("data/seed_docs_bert_200.npy", seed_docs)
    # np.save("data/seed_label_200.npy", seed_labels)
    # print(model_type)
    # if model_type=="cnn":
    #     print('here')
    #     return seed_docs["input_ids"], seed_labels, tokenizer
    return encoded_data, y_full, seed_docs, seed_labels

def tokenizeText(encoded_docs, vocabulary_inv_list):

    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # print(encoded_docs[0])
    # print(encoded_docs[1])

    # Convert to text, drop pads
    docs_text = []
    print("Converting encoding to text...")
    for doc in tqdm(encoded_docs):
        temp_doc = []
        for encoded_word in doc:
            if encoded_word == 0:
                continue
            temp_doc.append(vocabulary_inv[encoded_word])
        docs_text.append(" ".join(temp_doc))
    # print(docs_text)
    print("Converting text to tokens...")
    # texts = [tokenizer(text,padding='max_length', max_length = 450, 
    #                    truncation=True, return_tensors="pt") for text in tqdm(docs_text)]
    texts = tokenizer(docs_text, padding='max_length', max_length=100, truncation=True,)
    texts = {"input_ids": np.asarray(texts["input_ids"]), "attention_mask": np.asarray(texts["attention_mask"])}
    return texts

def build_vocab(docs):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*docs))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return word_counts, vocabulary, vocabulary_inv

def train_embed(sentence_matrix, vocab_inv, num_features=100, min_word_count=5, context=5):
    num_workers = 15  # Number of threads to run in parallel
    downsampling = 1e-3  # Downsample setting for frequent words
    print('Training Word2Vec model...')

    sentences = [[vocab_inv[w] for w in s] for s in sentence_matrix]
    sg = 1
    embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, sg=sg,
                                        vector_size=num_features, min_count=min_word_count,
                                        window=context, sample=downsampling)

    embedding_model.init_sims(replace=True)

    print(embedding_model)
    embedding_weights = {}
    for key, word in vocab_inv.items():
        if word in embedding_model.wv.key_to_index:
            value = embedding_model.wv[word]
        else:
            value = np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
        embedding_weights[key] = value
    # print(embedding_weights)
    return embedding_weights


def bertEmbeddingMat(dataset_name="agnews", train_embeddings=False):
    data_path = 'data/' + dataset_name
    data, y_full = read_file(data_path, True)

    # # Clean data
    # data = preprocess_doc(data)


    # Get subset of data for embeddings
    # data, y = reduce_data(data, y_full, 500, 4)
    print(len(data))

    # Get data statistics
    real_len, pseudo_len, embedding_len = get_statistics(data)
    print(real_len, pseudo_len)

    # Tokenize data
    print("Converting text to tokens...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_data = tokenizer(data, padding='max_length', max_length=real_len, truncation=True,)
    print("Done tokeniziging...")
    print(encoded_data.keys())

    # Build encoding dict
    word_counts, vocab, vocab_inv_list = build_vocab(encoded_data["input_ids"])
    vocab_inv = {key: value for key, value in enumerate(vocab_inv_list)}

    # Encode data:
    data = np.array([[vocab[word] for word in doc] for doc in encoded_data["input_ids"]])
    print(data.shape)

    if train_embeddings:
        embedding_weights = train_embed(data, vocab_inv)
        embedding_mat = np.array([np.array(embedding_weights[word]) for word in vocab_inv])
        np.save("data/embedding_mat_bert.npy", embedding_mat)
    else:
        embedding_mat = np.load("data/embedding_mat_bert.npy")

    return embedding_mat, data, vocab, vocab_inv


def bert_to_word2vec(docs, vocab):
    return np.array([[vocab[word] for word in doc] for doc in docs])
