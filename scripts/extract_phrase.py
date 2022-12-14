'''
Extract phrase from alignment file
'''
import argparse
import json
import torch

import sys
import string
import unicodedata
from stop_words import get_stop_words

import sys,re
import itertools
import fileinput
import math,copy
import random
import pdb

stopwords = dict.fromkeys(w for w in get_stop_words('en')+get_stop_words('de'))

tbl = dict.fromkeys(chr(i) for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

def is_not_punc_or_num(char):
    is_not_num = (re.search(r'[0-9]',char) is None)
    return (char not in tbl) and (is_not_num)

def is_punc(char):
    return (char in tbl)

def is_stop(char):
    is_num = (re.search(r'[0-9]',char) is not None)
    return (char in stopwords) or (is_num)

def phrase_extraction(srctext, trgtext, alignment, max_src_len=3):
    def extract(f_start, f_end, e_start, e_end):
        if f_end < 0:
            return {}
        tag=0
        for w in trgtext[f_start:f_end+1]:
            if is_punc(w):
                tag = 1
        if tag==1:
            return {}

        tag_stop=1
        for w in srctext[e_start:e_end+1]:
            if not is_stop(w):
                tag_stop=0
        if tag_stop==1:
            return {}

        for e,f in alignment:
            if ((f_start <= f <= f_end) and (e<e_start or e>e_end)):
                return {}

        phrases = set()
        fs = f_start
        fe = f_end
        tag_stop=1
        for w in trgtext[fs:fe+1]:
            if not is_stop(w):
                tag_stop=0
        if tag_stop==0:
            src_phrase = " ".join(srctext[i] for i in range(e_start,e_end+1))
            trg_phrase = " ".join([trgtext[i] for i in range(fs,fe+1) if i < len(trgtext)])
            phrases.add(((e_start,e_end+1), (fs,fe+1), src_phrase, trg_phrase))
        return phrases

    srctext = srctext.split()
    trgtext = trgtext.split()
    srclen = len(srctext)
    trglen = len(trgtext)

    e_aligned = [i for i,_ in alignment]
    f_aligned = [j for _,j in alignment]

    bp=set()
    for e_start in range(srclen):
        e_end = e_start+max_src_len-1
        if e_end > srclen-1:
            break
        if any([x not in e_aligned for x in range(e_start,e_end+1)]):
            continue
        tag = 0
        for w in srctext[e_start:e_end+1]:
            if is_punc(w):
                tag=1
        if tag==1:
            continue
        f_start,f_end = trglen-1,-1
        for e,f in alignment:
            if e_start <= e <= e_end:
                f_start = min(f,f_start)
                f_end = max(f,f_end)
        if any([x not in f_aligned for x in range(f_start,f_end+1)]):
            continue
        phrases = extract(f_start,f_end,e_start,e_end)
        
        if phrases:
            src_len = e_end - e_start
            tgt_len = f_end - f_start 
            if tgt_len < 1.6 * src_len +1 :
                bp.update(phrases)
    return sorted(bp)

def group_to_len(phrases):
    phr = [[] for i in range(3)]
    for w in phrases:
        phr[(w[0][1]-w[0][0])-1] += [w]
    return phr

def get_mapping(file_path):
    word_to_bpe=[]
    with open(file_path,"r", encoding='utf-8') as f:
        lines=f.readlines()
    for l in lines:
        subwords = l.strip('\n').split(' ')
        y = list(itertools.accumulate([int('@@' in x) for x in subwords]))
        x = [i-w+1 for i,w in enumerate(y)]
        x = [0]+x[:-1]
        y = [math.inf for i in range(x[-1]+1)]
        for i,w in enumerate(x):
            y[w] = int(min(i,y[w]))
        word_to_bpe += [y]
    return word_to_bpe

def main(args):
    stop_words=['@','<unk>','&quot','\'s','&apo','@-@','&quot;']
    dict_src = open(args.vocabname,'r').readlines()
    hf_words = [x.strip('\n').split(' ')[0] for x in dict_src[:100]]
    stop_words = stop_words + hf_words

    src=args.src
    trg=args.tgt
    consdir=args.raw+"/{}{}_{}_seed_{}".format(src,trg,args.contype, args.seed)
    random.seed(args.seed)

    with open(args.greedy,'r') as f:
        trans=f.readlines()
    if args.contype in ['talp']:
        with open(f'{args.raw}/test.talp.{src}','r') as f:
            valid_src=f.readlines()
        with open(f'{args.raw}/test.talp.{trg}','r') as f:
            valid_trg=f.readlines()
        with open(f'{args.raw}/test.talp.align','r') as f:
            aligns = f.readlines()
    elif args.contype in ['nst']:
        with open(f'{args.raw}/test.{src}','r') as f:
            valid_src=f.readlines()
        with open(f'{args.raw}/test.{trg}','r') as f:
            valid_trg=f.readlines()
        with open(f'{args.raw}/test.newstst.align','r') as f:
            aligns = f.readlines()
    else:
        pass

    word_to_bpe_src = get_mapping(f'{args.bpe}/test.{src}')
    word_to_bpe_trg = get_mapping(f'{args.bpe}/test.{trg}')

    all_cons={}
    total_cons=0
    
    for i, align in enumerate(aligns):
        align_int = []
        align=align.replace('p','-')
        for a in align.strip().split(' '):
            w=[int(s) for s in a.split('-')]  
            if args.reverse:
                w = [w[1],w[0]]
            align_int += [w]
        srctext=valid_src[i].strip()
        trgtext=valid_trg[i].strip()
        
        cons_num = random.sample([1,2,3],1)[0]
        word_nums = random.choices([1,2,3],k=cons_num)

        word_nums.sort(reverse=True)
        all_cons[str(i)]=[]
        restrict_posi=[]

        cons_dicts = [[],[],[]]
        for word_num in range(1, 3+1):
            phrase = phrase_extraction(srctext, trgtext, align_int, max_src_len=word_num)
            for item in phrase:
                src_cons = item[2]
                tgt_cons = item[3]
                src_start=word_to_bpe_src[i][item[0][0]]
                if item[0][1] < len(word_to_bpe_src[i]):
                    src_end=word_to_bpe_src[i][item[0][1]]
                else:
                    src_end=word_to_bpe_src[i][-1]+1

                trg_start=word_to_bpe_trg[i][item[1][0]]
                if item[1][1] < len(word_to_bpe_trg[i]):
                    trg_end=word_to_bpe_trg[i][item[1][1]]
                else:
                    trg_end=word_to_bpe_trg[i][-1]+1
                
                cons_dict = {}
                if tgt_cons not in trans[i] and all([is_not_punc_or_num(x) for x in src_cons.split(' ')]) and not all([x in stop_words for x in src_cons.split(' ')]):
                    cons_dict['src'] = src_cons
                    cons_dict['tgt'] = tgt_cons
                    cons_dict['src_span'] = [src_start,src_end-src_start]
                    cons_dict['tgt_span'] = [trg_start,trg_end-trg_start] 
                    cons_dicts[word_num].append(cons_dict)

        for word_num in word_nums:
            cnt_word_num = word_num - 1 
            if len(cons_dicts[cnt_word_num]) == 0:
                cnt_word_num = cnt_word_num - 1
                if len(cons_dicts[cnt_word_num]) == 0:
                    cnt_word_num = cnt_word_num - 1
                if cnt_word_num < 0:
                    break 
            random.shuffle(cons_dicts[cnt_word_num])
            for cons_dict in cons_dicts[cnt_word_num]:
                src_start = cons_dict['src_span'][0]
                src_end = cons_dict['src_span'][0] + cons_dict['src_span'][1]
                range_list = list(range(src_start,src_end))
                if all([x not in restrict_posi for x in range_list]):
                    all_cons[str(i)].append(cons_dict)
                    restrict_posi = restrict_posi + range_list
                    total_cons = total_cons + len(cons_dict['tgt'].split())
                    break 

    with open(f'{consdir}/constraints.{src}{trg}.{trg}.dict','w') as f:
        f.write(json.dumps(all_cons))
    print(f"finished all ...total cons is {total_cons}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="path for raw text and constraints")
    parser.add_argument("--bpe", required=False, help="path for bpe text")
    parser.add_argument("--src", required=True, help="source language")
    parser.add_argument("--tgt", required=False, help="target language")
    parser.add_argument("--vocabname", required=False, help="vocabulary for raw training text")
    parser.add_argument("--greedy", required=False, help="translations with greedy search")
    parser.add_argument("--contype", type=str, default='nst', required=False, help="which testset is used")
    parser.add_argument("--seed", required=False, default=1, type=int, help="seed to sample constraints")
    parser.add_argument("--reverse", action='store_true', help="set to reverse alignment compared to given reference alignment")
    args = parser.parse_args()
    main(args)

