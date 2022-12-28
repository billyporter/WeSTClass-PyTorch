#!/bin/bash

#$ -M wporter2@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu=1
#$ -N docs_5000_enc_data

# module load python/3.7.3
# module load pytorch/1.1.0
# module load cuda/11.2
# module load conda
# module load numpy
source /afs/crc.nd.edu/user/w/wporter2/anaconda3/bin/activate westclass3
conda env list
cd /afs/crc.nd.edu/user/w/wporter2/WeSTClass-PyTorch/
python3 main.py --model bert --data generate
