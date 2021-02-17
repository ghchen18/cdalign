#!/bin/bash

Calculate_AER_scores(){   
    fseq=$1
    src=$2
    tgt=$3
    task=$4
    split=$5 

    resdir=$fseq/genres/${src}2${tgt}_${task} && mkdir -p $resdir 
    if [[ $task == *"attin"* ]] ; then
        set_shift='--set-shift' 
        LayerNum=2 
        ModelName=vanilla    
    elif [[ $task == *"attout"* ]] ; then
        set_shift='' 
        LayerNum=4  
        ModelName=vanilla  
    else
        set_shift='' 
        LayerNum=4  
        ModelName=eamout 
    fi

    if [[ $split == *"test"* ]] ; then
        extract='/talp'
    else
        extract=''
    fi

    echo "Extract alignment from src-to-tgt Transformer"
    Modeldir=$fseq/models/${src}2${tgt}_${ModelName}
    python generate_align.py ${fseq}${extract} -s $src -t $tgt --path $Modeldir/checkpoint_best.pt \
        --max-tokens 6000 --beam 1 --remove-bpe --print-vanilla-alignment --alignment-task $task \
        --decoding-path $resdir --quiet --gen-subset $split --model-overrides "{'alignment_layer': $LayerNum}" ${set_shift}

    echo "Extract alignment from tgt-to-src Transformer"
    Modeldir=$fseq/models/${tgt}2${src}_${ModelName}
    python generate_align.py $fseq${extract} -s $tgt -t $src --path $Modeldir/checkpoint_best.pt \
        --max-tokens 6000 --beam 1 --remove-bpe --print-vanilla-alignment --alignment-task $task \
        --decoding-path $resdir --quiet --gen-subset $split --model-overrides "{'alignment_layer': $LayerNum}" ${set_shift}

    if [[ $split == *"test"* ]] ; then
        echo "Start to extract alignment for testsets of alignment task."
        ## transfer alignment for BPE tokens to alignments for raw texts without BPE
        python scripts/aer/sentencepiece_to_word_alignments.py --src $fseq/talp/bpe/$split.$src \
            --tgt $fseq/talp/bpe/$split.$tgt --input $resdir/$split.${src}2${tgt}.align --output $resdir/$split.${src}2${tgt}.raw.align
        python scripts/aer/sentencepiece_to_word_alignments.py --src $fseq/talp/bpe/$split.$tgt \
            --tgt $fseq/talp/bpe/$split.$src --input $resdir/$split.${tgt}2${src}.align --output $resdir/$split.${tgt}2${src}.raw.align
        echo "start to merge src-to-tgt and tgt-to-src alignments ..."
        python scripts/aer/combine_bidirectional_alignments.py $resdir/$split.${src}2${tgt}.raw.align  $resdir/$split.${tgt}2${src}.raw.align --method 'grow-diagonal' > $resdir/$split.${src}2${tgt}.bidir.align

        ref_align=$fseq/raw/test.talp.align
        echo "=====AER is start to calculate AER for de-en layer=${LayerNum}..."   
        python scripts/aer/aer.py ${ref_align} $resdir/$split.de2en.raw.align --fAlpha 0.5 --oneRef
        python scripts/aer/aer.py ${ref_align} $resdir/$split.en2de.raw.align --fAlpha 0.5 --oneRef --reverseHyp
        python scripts/aer/aer.py ${ref_align} $resdir/$split.de2en.bidir.align --fAlpha 0.5 --oneRef
    else
        echo "start to merge src-to-tgt and tgt-to-src alignments of BPE tokens for training EAM module"
        python scripts/aer/combine_bidirectional_alignments.py $resdir/$split.${src}2${tgt}.align  $resdir/$split.${tgt}2${src}.align --method 'grow-diagonal' > $fseq/bpe/$split.align 
    fi 
}

set -e 
cd ..
cd ..
export CUDA_VISIBLE_DEVICES=0
fseq=/path/to/processed/fairseq/data  ## fairseq processed data with shared embeddings 

src=de 
tgt=en 

echo "Task 1: Calculate AER scores on alignment testset"
task=attin ## choices = ['attin', 'eamout','attout']
Calculate_AER_scores $fseq $src $tgt $task 'test'

echo "Task 2: extract alignment with Att-Input and process the alignments on the training set and valid set"
task=attin 
Calculate_AER_scores $fseq $src $tgt $task 'valid'
Calculate_AER_scores $fseq $src $tgt $task 'train'

echo "Start processing alignment data into fairseq data format"
python preprocess.py -s $src -t $tgt --dataset-impl lazy \
    --workers 8 --destdir $fseq --align-suffix align --joined-dictionary  \
    --trainpref $fseq/bpe/train --validpref $fseq/bpe/valid \
    --srcdict $fseq/bak/dict.${src}.txt --process-only-alignment 
    