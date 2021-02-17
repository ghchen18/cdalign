import re,argparse
import os,json
import time,random
import itertools

def cal_csr(args):
    with open(args.tgt_cons,'r') as load_f:
        tgt_pt = json.load(load_f)
    total_count = 0
    copy_count = 0
    with open(args.hypo,'r') as fin:
        for idx, line in enumerate(fin):
            if str(idx) in tgt_pt:
                for phrase in tgt_pt[str(idx)]:
                    if 'tgt' in phrase:
                        tgt_word = phrase['tgt']
                    elif 'tgt_phrase' in phrase:
                        tgt_word = phrase['tgt_phrase']
                        
                    tgt_len = len(tgt_word.split())
                    total_count = total_count + tgt_len
                    if re.search(r'\b{}\b'.format(tgt_word), line) is not None:
                        copy_count = copy_count + tgt_len
                    elif ' '+tgt_word+' ' in line:
                        copy_count = copy_count + tgt_len
                    elif re.search(r'^{}'.format(tgt_word+' '), line):
                        copy_count = copy_count + tgt_len
                    elif re.search(r'{}$'.format(' '+tgt_word), line):
                        copy_count = copy_count + tgt_len
    csr=copy_count/total_count
    print(f"Total number of given constraints is {total_count}, Successfully generated constraints number is {copy_count}, CSR={csr}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_cons", required=False, help="original source and target sentences.")
    parser.add_argument("--hypo", required=False, help="output log and result.")
    args = parser.parse_args()

    cal_csr(args)
