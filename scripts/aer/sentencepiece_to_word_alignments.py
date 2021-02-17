#!/usr/bin/env python3
# code is from <https://github.com/lilt/alignment-scripts>
import sys,pdb
import itertools
import fileinput
import argparse

def get_mapping(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for l in f:
            subwords = l.strip().split()  
            tmp = [0] + list(itertools.accumulate([int('@@' in x) for x in subwords]))[:-1]
            orig_range = list(itertools.accumulate([int(True) for x in subwords]))
            result = [ a-b-1 for a,b in zip(orig_range,tmp)]  ## -1 then index start at 0.

            yield result

def convert(args):
    src_file = args.src
    tgt_file = args.tgt
    bpe_alignments = open(args.input,'r').readlines()
    examples = zip(get_mapping(src_file), get_mapping(tgt_file), bpe_alignments)
    
    for src_map, tgt_map, line in examples:
        subword_alignments = {(int(a), int(b)) for a, b in (x.split("-") for x in line.split())}       
        word_alignments = {"{}-{}".format(src_map[a], tgt_map[b]) for a, b in subword_alignments}
        yield word_alignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="src bpe text")
    parser.add_argument("--tgt", required=False, help="tgt bpe text")
    parser.add_argument("--input", required=True, help="alignment after bpe")
    parser.add_argument("--output", required=False, help="alingment in text without bpe")
    args = parser.parse_args()
    '''
    Note:the input and output alignment requires: index start at 0.

    '''

    with open(args.output,'w') as fout:
        for word_alignment in convert(args):
            # print(" ".join(word_alignment))
            fout.write(" ".join(word_alignment)+'\n')

