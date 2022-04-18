from transformers import BertTokenizer


def tokenizeText(encoded_docs, vocabulary_inv_list):
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    docs_text = []
    for doc in encoded_docs:
        temp_doc = []
        for sent in doc:
            temp_sent = []
            for word in sent:
                temp_sent.append(vocabulary_inv[word])
            temp_doc.append(temp_sent)
        docs_text.append(temp_doc)

    
    seed_text = []
    for doc in docs_text:
        local_str = ''
        for i, sent in enumerate(doc):
            local_str += ' '.join(sent[0:15])
            if i == 2:
                break
        seed_text.append(local_str)

    texts = [tokenizer(text,padding='max_length', max_length = 128, 
                       truncation=True, return_tensors="pt") for text in seed_text]
    return texts