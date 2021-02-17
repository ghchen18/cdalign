#!/bin/bash

train_EAM_output(){   
    fseq=$1
    src=$2
    tgt=$3
    modeldir=$fseq/models/${src}2${tgt}_eamout && mkdir -p $modeldir 
    MaxUpdates=10000 
    cp $fseq/models/${src}2${tgt}_vanilla/checkpoint_best.pt $modeldir/checkpoint_last_f.pt 

    python train.py $fseq \
        -a transformer_wmt_en_de --optimizer adam --lr 0.0005 -s $src -t $tgt \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000  --seed 32 --fp16 \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0 --activation-fn relu \
        --criterion label_smoothed_cross_entropy_with_alignment --max-update $MaxUpdates \
        --warmup-updates 4000 --warmup-init-lr '1e-07' --keep-last-epochs 50  \
        --adam-betas '(0.9, 0.98)' --save-dir $modeldir --update-freq 8 --ddp-backend no_c10d --share-all-embeddings \
        --alignment-task 'eamout'  --alignment-layer 4  --load-alignments 
}

train_vanilla_transformer(){   
    fseq=$1
    src=$2
    tgt=$3
    modeldir=$fseq/models/${src}2${tgt}_vanilla && mkdir -p $modeldir 
    MaxUpdates=100000 

    python train.py $fseq \
        -a transformer_wmt_en_de --optimizer adam --lr 0.0005 -s $src -t $tgt \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000  --fp16 \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0 --activation-fn relu \
        --criterion label_smoothed_cross_entropy --max-update $MaxUpdates \
        --warmup-updates 4000 --warmup-init-lr '1e-07' --keep-last-epochs 50 \
        --adam-betas '(0.9, 0.98)' --save-dir $modeldir --update-freq 8 --ddp-backend no_c10d \
        --share-all-embeddings --alignment-task 'vanilla' 
}

set -e 
cd ..

export CUDA_VISIBLE_DEVICES=0
fseq=/path/to/processed/fairseq/data  ## fairseq processed data with shared embeddings 

echo "Task 1: Train vanilla Transformer model"
train_vanilla_transformer $fseq de en 
train_vanilla_transformer $fseq en de 

echo "Task 2: Train EAM-Output model with the params of vanilla Transformer frozen"
train_EAM_output $fseq de en 
train_EAM_output $fseq en de 
