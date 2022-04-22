from transformers import BertTokenizer
from tqdm import tqdm

def tokenizeText(encoded_docs, vocabulary_inv_list):
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Convert to text, drop pads
    docs_text = []
    print("Converting encoding to text...")
    for doc in tqdm(encoded_docs):
        temp_doc = []
        for sent in doc:
            for encoded_word in sent:
                if encoded_word == 0:
                    continue
                temp_doc.append(vocabulary_inv[encoded_word])
        docs_text.append(" ".join(temp_doc))
    # print(docs_text)
    print("Converting text to tokens...")
    texts = [tokenizer(text,padding='max_length', max_length = 450, 
                       truncation=True, return_tensors="pt") for text in tqdm(docs_text)]
    return texts