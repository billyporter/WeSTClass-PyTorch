import numpy as np
import torch
import torch.nn as nn
import warnings
import os
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
import spherecluster
from spherecluster import SphericalKMeans, VonMisesFisherMixture, sample_vMF
from transformers import BertTokenizer

def psuedodocs_bert(embedding_mat, encode_count_dict, keyword_encodes, encode_list, centers, kappas, doc_settings):

    # Constants TODO (@billy): Clean up later
    num_doc_per_class, total_num, interp_weight, doc_len, sequence_length = doc_settings
    num_class = len(keyword_encodes)

    docs = np.zeros((num_doc_per_class * num_class, doc_len), dtype='int32')
    labels = np.zeros((num_doc_per_class * num_class, num_class))

    # Create background of word frequency
    background_array = np.zeros(len(encode_list))
    total_count = 0
    for i in range(0, len(encode_list)):
        current_encode = encode_list[i]
        current_count = encode_count_dict[current_encode]
        total_count += current_count
        background_array[i] = current_count
    background_array = np.true_divide(background_array, total_count)
    background_vec = interp_weight * background_array

    # Loop over classes
    for i in range(0, len(keyword_encodes)):
        center, kappa = centers[i], kappas[i]
        # print(center)
        # print(kappa)
        discourses = sample_vMF(center, kappa, num_doc_per_class)

        # Loop over discourses
        for j in range(num_doc_per_class):
            discourse = discourses[j]
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Get Similarity
            vocab_keyword_similarity = np.dot(embedding_mat, discourse)

            ### Remove Duplicates ###
            # First pass to get max of each encoding
            encoding_max_dict = {}
            for index, sim in enumerate(vocab_keyword_similarity):
                encoding = encode_list[index]
                if encoding not in encoding_max_dict:
                    encoding_max_dict[encoding] = (index, sim)
                else:
                    if sim > encoding_max_dict[encoding][1]:
                        encoding_max_dict[encoding] = (index, sim)

            # Second pass to zero out the non maxes:
            for index in range(0, len(vocab_keyword_similarity)):
                encoding = encode_list[index]
                if index != encoding_max_dict[encoding][0]:
                    vocab_keyword_similarity[index] = 0.0
            ### End Remove Duplicates ###
            
            # Sort by similarity
            similar_indices = sorted(range(len(vocab_keyword_similarity)), key=lambda k: vocab_keyword_similarity[k], reverse=True)[:sequence_length]

            # Restrict vocabulary
            vocab_keyword_similarity = np.exp(vocab_keyword_similarity)
            sorted_idx = np.argsort(vocab_keyword_similarity)[::-1]
            delete_idx = sorted_idx[sequence_length:]
            vocab_keyword_similarity[delete_idx] = 0
            vocab_keyword_similarity /= np.sum(vocab_keyword_similarity)

            # Adjust vocabulary for word frequency
            vocab_keyword_similarity *= 1 - interp_weight
            vocab_keyword_similarity += background_vec

            similar_indices = np.random.choice(len(vocab_keyword_similarity), size=sequence_length, p=vocab_keyword_similarity, replace=False)
            similar_encodings = encode_list[similar_indices]
            # print("Word: ", tokenizer.decode(similar_encodings))
            docs[i*num_doc_per_class+j][:sequence_length] = similar_encodings 
            labels[i*num_doc_per_class+j] = interp_weight/num_class*np.ones(num_class)
            labels[i*num_doc_per_class+j][i] += 1 - interp_weight
    return docs, labels


def label_expansion_bert(embedding_mat, embed_encode_dict, keyword_embeds, keyword_encodes, tokenizer):
    print("Retrieving top-t nearest words...")

    # Get Average Embedding of keywords
    avg_embeddings = []
    # embedding_mat = sorted(embedding_mat)
    for class_embeds in keyword_embeds:
        word_embeddings = torch.zeros((len(class_embeds), 768))
        for j, embed in enumerate(class_embeds):
            word_embeddings[j] = torch.FloatTensor(embed)
        combined_embedding = torch.mean(word_embeddings, 0)
        avg_embeddings.append(combined_embedding)

    #### Generate psuedo document vocabulary ####
    words_list = []
    all_class_labels = []
    sz = 0
    # Stop when an embedding is shared between classes
    while len(all_class_labels) == len(set(all_class_labels)):
        sz += 1
        expanded_array, exp_debug, debug2 = seed_expansion_bert(sz, embedding_mat, avg_embeddings, embed_encode_dict, keyword_embeds, keyword_encodes)
        all_class_labels = [w for w_class in debug2 for w in w_class]
        all_class_debugs = [w for w_class in exp_debug for w in w_class]
    # expanded_array, exp_debug = seed_expansion_bert(sz - 1, embedding_mat, avg_embeddings, embed_encode_dict, keyword_embeds, keyword_encodes)

    centers = []
    kappas = []
    print("Top-t nearest words for each class:")
    for i in range(len(exp_debug)):
        expanded_mat = expanded_array[i]
        expanded_encodes = exp_debug[i]
        vocab_expanded = tokenizer.decode(expanded_encodes)
        print("Class {}:".format(i))
        print(vocab_expanded)
        # break
        vmf_soft = VonMisesFisherMixture(n_clusters=1, n_jobs=15)
        vmf_soft.fit(expanded_mat)
        center = vmf_soft.cluster_centers_[0]
        kappa = vmf_soft.concentrations_[0]
        centers.append(center)
        kappas.append(kappa)

    print("Finished vMF distribution fitting.")
    return expanded_array, centers, kappas


def seed_expansion_bert(sz, embedding_mat, avg_embeddings, embed_encode_dict, keyword_embeds, keyword_encodes):
    expanded_seed_debug = []
    expanded_seed = []
    expanded_seed_debug2 = []
    # embedding_mat = sorted(embeddin)
    for i in range(0, len(keyword_embeds)):

        # Get most similar words to keywords
        expanded = np.dot(embedding_mat, avg_embeddings[i])
        word_expanded = sorted(range(len(expanded)), key=lambda k: expanded[k], reverse=True)
        expanded_class_encodings = keyword_encodes[i] + []
        expanded_class_embeds = keyword_embeds[i] + []
        words_added = 0
        word_index = 0

        # Add most similar keywords until size reached
        while words_added < sz:
            embedding_index = word_expanded[word_index]
            embedding = tuple(embedding_mat[embedding_index].tolist())
            encoding = embed_encode_dict[embedding]
            if encoding not in expanded_class_encodings:
                expanded_class_encodings.append(encoding)
                expanded_class_embeds.append(embedding)
                words_added += 1
            word_index += 1
        expanded_seed_debug.append(np.array(expanded_class_encodings))
        expanded_seed.append(np.array(expanded_class_embeds))
        expanded_seed_debug2.append(expanded_class_embeds)
    return expanded_seed, expanded_seed_debug, expanded_seed_debug2