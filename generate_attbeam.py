#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
constrain with mask prob change
"""

import torch

from fairseq import bleu, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_attbeam import SequenceAttbeam
from fairseq.utils import import_user_module
import json,os

def string_clean(hypo_str):
    special_tokens = ['<pad>','</s>']
    hypo_str_new = []
    for w in hypo_str.strip().split(' '):
        if w in special_tokens:
            continue
        hypo_str_new += [w]
    hypo_str = ' '.join(hypo_str_new)
    return hypo_str


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    from fairseq import checkpoint_utils
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'), task=task, arg_overrides=eval(args.model_overrides),
    )

    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()

    translator = SequenceAttbeam(
        models, task.source_dictionary, task.target_dictionary, beam_size=args.beam, minlen=args.min_len,
        stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen, unk_penalty=args.unkpen,
        sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.temperature,
        diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength, args=args,
    )

    if use_cuda:
        translator.cuda()

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True

    def to_tokens(token_id, my_dict, escape_unk=False):
        tokens = []
        for w in token_id:
            if w == 1:
                continue
            else:
                if w == my_dict.unk():
                    tokens.append(my_dict.unk_string(escape_unk))
                else:
                    tokens.append(my_dict[w])
        return tokens

    with open(args.trg_pt,'r') as load_f:
        trg_pt = json.load(load_f)
    with progress_bar.build_progress_bar(args, itr) as t:
        if args.decoding_path is not None:
            src_sents = [[] for _ in range(10000)]
            tgt_sents = [[] for _ in range(10000)]
            hyp_sents = [[] for _ in range(10000)]

        translations = translator.generate_batched_itr(
            t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
            cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
            trg_pt=trg_pt, tgt_dict=tgt_dict,
        )
        wps_meter = TimeMeter()  
        for hypoID, src_tokens, target_tokens, hypo in translations:
            index = (hypo ==  tgt_dict.eos()).nonzero()

            if index.size(0) > 1:
                index = index[0,0]
                hypo = hypo[index+1:]

            hypo_align = to_tokens(hypo,tgt_dict)
            ref_align = to_tokens(target_tokens[0], tgt_dict, escape_unk=True)
            src_align = to_tokens(src_tokens[0], src_dict)

            src_str = ' '.join(src_align)
            target_str = ' '.join(ref_align)
            hypo_str = ' '.join(hypo_align)

            src_str = src_dict.string(src_tokens[0], args.remove_bpe, escape_pad=True, tgt_dict=tgt_dict)
            target_str = tgt_dict.string(target_tokens[0], args.remove_bpe, escape_unk=True,escape_pad=True)
            hypo_str = tgt_dict.string(hypo, args.remove_bpe, escape_unk=True, escape_pad=True)

            if hasattr(scorer, 'add_string'):
                scorer.add_string(target_str, hypo_str)
            else:
                target_tokens = target_tokens.int()
                hypo = hypo.int().cpu()
                scorer.add(target_tokens, hypo)

            wps_meter.update(src_tokens.size(0))
            num_sentences += 1

            if args.decoding_path is not None:
                src_sents[int(hypoID)].append(src_str)
                tgt_sents[int(hypoID)].append(target_str)
                hyp_sents[int(hypoID)].append(hypo_str)

    print('| >=>=>=> Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

    if args.decoding_path is not None:
        with open(os.path.join(args.decoding_path, 'source.txt'), 'w') as f:
            for sents in src_sents:
                if len(sents)==0:
                    continue
                for sent in sents:
                    f.write(sent+'\n')

        with open(os.path.join(args.decoding_path, 'target.txt'), 'w') as f:
            for sents in tgt_sents:
                if len(sents)==0:
                    continue
                for sent in sents:
                    f.write(sent+'\n')

        with open(os.path.join(args.decoding_path, 'decoding.txt'), 'w') as f:
            for sents in hyp_sents:
                if len(sents)==0:
                    continue
                for sent in sents:
                    f.write(sent+'\n')



def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == '__main__':
    cli_main()
