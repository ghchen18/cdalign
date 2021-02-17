#!/bin/bash
test_clean(){   
    src=$1
    tgt=$2
    fseq=$3
    mode=$4
    modeldir=$fseq/models/${src}2${tgt}_vanilla
    raw_reference=$fseq/raw/test.$tgt 
    
    if [[ $mode == *"greedy"* ]] ; then
        echo "Translate with greedy search for constraints extraction"
        resdir=$fseq/genres/${src}2${tgt}_greedy && mkdir -p $resdir
        beamsize=1    
    else
        resdir=$fseq/genres/${src}2${tgt}_vanilla && mkdir -p $resdir
        beamsize=5     
    fi

    python generate.py $fseq -s $src -t $tgt \
        --path $modeldir/checkpoint_best.pt \
        --batch-size 100 --beam $beamsize  --sacrebleu \
        --decoding-path $resdir --quiet --remove-bpe

    detok=/path/to/moses/scripts/detokenize.perl
    perl $detok -l $tgt < $resdir/decoding.txt  > $resdir/decoding.detok 
    perl $detok -l $tgt < $raw_reference > $resdir/target.detok
    cat $resdir/decoding.detok | sacrebleu $resdir/target.detok
}

extract_cons(){   
    src=$1
    tgt=$2
    fseq=$3
    bpe=$fseq/bpe 
    test_clean $src $tgt $fseq greedy
    if [[ $tgt == *"de"* ]] || [[ $tgt == *"ro"* ]] ; then
        reverse='--reverse'
    else
        reverse=''
    fi
    python scripts/extract_phrase.py --raw $fseq/raw --src $src --tgt $tgt \
        --bpe $bpe --greedy $fseq/genres/${src}2${tgt}_greedy/decoding.txt \
        --vocabname $fseq/raw/vocab.$src $reverse
}


test_on_constrained_decoding_task(){   
    src=$1
    tgt=$2
    fseq=$3
    modeldir=$4
    task=$5

    raw_reference=$fseq/raw/test.$tgt
    consdir=$fseq/raw/${src}${tgt}_nst_seed_1 && mkdir -p $consdir
    resdir=$fseq/genres/${task}_${src}${tgt}_seed_1 && mkdir -p $resdir

    if [[ $task == *"attin"* ]] ; then
        set_shift='--set-shift' 
        Layer=2    
    else
        set_shift='' 
        Layer=4      
    fi

    python generate_attbeam.py $fseq -s $src -t $tgt \
        --path ${modeldir}/checkpoint_best.pt --decoding-path $resdir \
        --batch-size 1 --beam 5 --sacrebleu \
        --trg_pt $consdir/constraints.${src}${tgt}.${tgt}.dict --remove-bpe  \
        --alignment-layer $Layer --alignment-task $task ${set_shift} \
        --model-overrides "{'alignment_layer': $Layer}"

    detok=/path/to/moses/scripts/detokenize.perl
    perl $detok -l $tgt < $resdir/decoding.txt  > $resdir/decoding.detok 
    perl $detok -l $tgt < $raw_reference > $resdir/target.detok
    cat $resdir/decoding.detok | sacrebleu $resdir/target.detok
    python scripts/Calculate_CSR.py  --tgt_cons $consdir/constraints.${src}${tgt}.${tgt}.dict --hypo $resdir/decoding.txt 
}



set -e 
cd ..

export CUDA_VISIBLE_DEVICES=0
fseq=/path/to/processed/fairseq/data  ## fairseq processed data with shared embeddings 
src=de 
tgt=en 

echo "Task 1: Test on clean testset without constraints"
test_clean $src $tgt $fseq beam_5

echo "Task 2: Extract constraint pairs from sentence pairs"
extract_cons $src $tgt $fseq

echo "Task 3: Constrained decoding with ATT-Output method"
task=attout
modeldir=$fseq/models/${src}2${tgt}_vanilla
test_on_constrained_decoding_task $src $tgt $fseq $modeldir $task

echo "Task 4: Constrained decoding with ATT-Input method"
task=attin
modeldir=$fseq/models/${src}2${tgt}_vanilla
test_on_constrained_decoding_task $src $tgt $fseq $modeldir $task

echo "Task 5: Constrained decoding with EAM-Output method"
task=eamout
modeldir=$fseq/models/${src}2${tgt}_eamout 
test_on_constrained_decoding_task $src $tgt $fseq $modeldir $task
