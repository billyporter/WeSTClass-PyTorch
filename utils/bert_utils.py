from base64 import decode, encode
from tkinter.filedialog import test
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
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


    return real_len, pseudo_len

def extract_keywords(data_path, num_keywords, data, perm):
    sup_data = []
    sup_idx = []
    sup_label = []
    file_name = 'doc_id.txt'
    infile = open(join(data_path, file_name), mode='r', encoding='utf-8')
    text = infile.readlines()
    print(data[0])
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
    print(sup_data)

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

def load_data_bert(dataset_name="agnews", sup_source="keywords", with_evaluation=True, gen_seed_docs="generate", batch_size=8):
    data_path = 'data/' + dataset_name
    data, y = read_file(data_path, with_evaluation)

    # Clean data
    data = preprocess_doc(data)

    # Get data statistics
    real_len, pseudo_len = get_statistics(data)
    print(real_len, pseudo_len)

    # Add keywords to end sentence
    if sup_source in ("labels", "keywords"):
        keywords = load_keywords(data_path, sup_source)
    elif sup_source == "docs":
        sz = len(data)
        np.random.seed(1234)
        perm = perm = np.random.permutation(sz)
        data_copy = [s.split(" ") for s in data]
        keywords, sup_idx = extract_keywords(data_path, 10, data_copy, perm)
    keywords_sents = [" ".join(sent) for sent in keywords]
    print(keywords)
    print('Done preprocessing...')

    # Tokenize data
    print("Converting text to tokens...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_data = tokenizer(data, padding='max_length', max_length=real_len, truncation=True,)
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
    # bert_data = BertDataWrapper(encoded_data)
    bert_loader = DataLoader(dataset=bert_data, batch_size=batch_size, shuffle=False)

    # Keywords model prep
    tokens_tensor_keywords = torch.tensor(encoded_keywords["input_ids"])
    segments_tensor_keywords = torch.tensor(encoded_keywords["token_type_ids"])

    # Construct embedding: id dictionary
    embed_encode_dict = {}
    encode_count_dict = {}
    keyword_embeds = []
    keyword_encodes = []

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

            # Add embeddings to dictionary
            token_embeddings = torch.squeeze(last_hidden_state, dim=1)
            for i, document in enumerate(tokens_tensor):
                for j, encoding in enumerate(document):
                    word = tokenizer.decode(encoding)
                    if word == "[ C L S ]":
                        continue
                    if word == "[ S E P ]":
                        continue
                    if word == "[ P A D ]":
                        break
                    embedding = token_embeddings[i][j].cpu().numpy()
                    embedding = tuple(embedding / np.linalg.norm(embedding))
                    encoding = encoding.cpu().numpy().item()
                    embed_encode_dict[embedding] = encoding
                    encode_count_dict[encoding] = encode_count_dict.get(encode, 0) + 1
        
        # Keywords
        if torch.cuda.is_available():
            tokens_tensor_keywords = tokens_tensor_keywords.cuda()
            segments_tensor_keywords = segments_tensor_keywords.cuda()
        
        outputs = model(tokens_tensor_keywords, segments_tensor_keywords)
        hidden_states = outputs[2]
        last_hidden_state = hidden_states[-2]
        token_embeddings = torch.squeeze(last_hidden_state, dim=1)
        for i, document in enumerate(tokens_tensor_keywords):
            class_embeds = []
            class_encodes = []
            for j, encoding in enumerate(document):
                word = tokenizer.decode(encoding)
                if word == "[ C L S ]":
                    continue
                if word == "[ S E P ]":
                    continue
                if word == "[ P A D ]":
                    break
                embedding = tuple(token_embeddings[i][j].tolist())
                encoding = encoding.cpu().numpy().item()
                class_embeds.append(embedding)
                class_encodes.append(encoding)
            keyword_embeds.append(class_embeds)
            keyword_encodes.append(class_encodes)
    print("Finished Embedding Generation...")

    seed_docs_input_ids = []
    seed_labels = []
    if gen_seed_docs == "generate":
        # Create embedding matrix
        embedding_mat = np.array(list(embed_encode_dict.keys()))
        encode_list = np.array(list(embed_encode_dict.values()))

        # Get VMF Distribution
        _, centers, kappas = label_expansion_bert(embedding_mat, embed_encode_dict, keyword_embeds, keyword_encodes, tokenizer)

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
    
    return encoded_data, y, seed_docs, seed_labels