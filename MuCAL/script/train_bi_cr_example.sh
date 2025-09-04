#!/bin/bash


nvcc --version
nvidia-smi

# For training and evaluation of BiEncoder
python contrastive/BiEncoder/main.py \
-ep 10 \
-tbs 16 \
-vbs 32 \
-lp 'ckpt/load_a_trained_model_or_pretrained_model' \
-sp 'ckpt/new_model_to_save' \
-id 'data/dev/dev_data.json' \
-ie 'data/test/test_data.json' \
-langs 'en,zh,ru,ar,fr,es' \
-tr \
-hn \
-hn_num 2 \
-eval \
-wandb


# For training and evaluation of CrossEncoder
python contrastive/CrossEncoder/main.py \
-ep 10 \
-tbs 16 \
-vbs 32 \
-lp 'ckpt/load_a_trained_model_or_pretrained_model' \
-sp 'ckpt/new_model_to_save' \
-id 'data/dev/dev_data.json' \
-ie 'data/test/test_data.json' \
-langs 'en,zh,ru,ar,fr,es' \
-tr \
-eval \
-wandb
