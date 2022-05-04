# WeSTClass-PyTorch
PyTorch Implementation of WeSTClass
Weakly-Supervised Neural Text Classification
[http://chaozhang.org/papers/2018-cikm-westclass.pdf](http://chaozhang.org/papers/2018-cikm-westclass.pdf)



## Usage               
usage: main.py [-h] [--data {generate,load}] [--model {rnn,bert,cnn}]\
               &emsp; &emsp; &emsp; [--sup_source {labels,keywords,docs}] [--pretrain]\
               &emsp; &emsp; &emsp; [--selftrain] [--evaluate] [--with_statistics] [--load_model]\
               &emsp; &emsp; &emsp; [--save_docs]

optional arguments:\
  &nbsp; -h, --help            show this help message and exit\
  &nbsp; --data {generate,load}\
  &nbsp; --model {rnn,bert,cnn}\
  &nbsp; --sup_source {labels,keywords,docs}\
  &nbsp; --pretrain\
  &nbsp; --selftrain\
  &nbsp; --evaluate\
  &nbsp; --with_statistics\
  &nbsp; --load_model\
  &nbsp; --save_docs


## Requirements
- Numpy Version: 1.21.5
- PyTorch Version: 1.10.0+cu111
- Python Version: 3.7.12
- Scikit-Learn Version: 0.22
- TQDM Version: 4.63.0
- gensim Version: 4.2.0
