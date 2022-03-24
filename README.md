# WeSTClass-PyTorch
PyTorch Implementation of WeSTClass
Weakly-Supervised Neural Tex Classification
[http://chaozhang.org/papers/2018-cikm-westclass.pdf](http://chaozhang.org/papers/2018-cikm-westclass.pdf)


## General Summary
### Pseudo Document Generation
Given a corpus containing documents with N classes, we aim to generate pseudo-documents to be used for model pretraining. First, we construct word embeddings for the entire voacbulary of words. Next, we use a few user provided keywords for each class and select the top-t words that have embeddings most similar to the key words. "t" is constructed to be the maximum integer such that the classes do not have any overlaps in words. With the selected keywords, we fit the embeddings to a unit sphere and repeatedly sample the words to generate pseudo documents (~500 per class). Finally, we construct loosely fitted pseudo labels (for 4-classes, we use a label of 0.85 for the class the psuedo document belongs to and 0.05 for the other three classes, rather than using a one hot encoding in order to avoid the neural model from overfittting on the pseudo documents

### Neural Model
Given the pseudo documents and their pseudo labels, we pretrain a Hierarchical Attention Network (HAN) based on word and sentence level attention with a RNN. After pre-training the model, we perform a self training loop that functions as follows. First, we make class predictions for each real document. For every iteration, we train the neural model on a batch of the real documents coupled with labels computed from a common self-training formula based on the soft frequency of each class. After every N iterations, we re-predict the classes and evaluate what percentage of the predictions changes. If that percentage is smaller than a given threshold, we terminate self-training.


## Usage
usage: main.py [-h] [-m {p,s,e}] [--batch_size BATCH_SIZE] [--maxiter MAXITER]\
               &emsp; &emsp; &emsp; [--pretrain_epochs PRETRAIN_EPOCHS]\
               &emsp; &emsp; &emsp; [--update_interval UPDATE_INTERVAL]

optional arguments:\
  &nbsp; -h, --help            &emsp; &emsp; show this help message and exit\
  &nbsp; -m {p,s,e}, --method {p,s,e}\
                        &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; p: pretrain\
                        &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; s: self-train\
                        &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; e: eval\
  &nbsp; --batch_size BATCH_SIZE\
  &nbsp; --maxiter MAXITER\
  &nbsp; --pretrain_epochs PRETRAIN_EPOCHS\
  &nbsp; --update_interval UPDATE_INTERVAL\
 &nbsp;  --data {full, trunc}, DATA \
                        &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; full: full 120,000 documents\
                        &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; trunc: 2,000 documents


## Requirements
- Numpy Version: 1.21.5
- PyTorch Version: 1.10.0+cu111
- Python Version: 3.7.12
- TQDM Version: 4.63.0
