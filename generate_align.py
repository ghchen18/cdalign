#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch,os

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

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
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())

    if args.print_vanilla_alignment:
        import string
        punc = string.punctuation
        src_punc_tokens = [w for w in range(len(src_dict)) if src_dict[w] in punc]
    else:
        src_punc_tokens = None

    with progress_bar.build_progress_bar(args, itr) as t:
        if args.decoding_path is not None:
            if args.gen_subset == 'train':
                align_sents = [[] for _ in range(5000000)]
            else:
                align_sents = [[] for _ in range(10000)]

        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue
            sample_id = sample['id'].tolist()[0]
            if args.print_vanilla_alignment:
                if not args.set_shift:
                    alignments = utils.extract_att_output_alignment(sample, models[0], src_punc_tokens, alignment_task=args.alignment_task)
                else:  
                    alignments = utils.extract_att_input_alignment(sample, models[0], src_punc_tokens, alignment_task=args.alignment_task)
            else:
                alignments = None
            
            for i, sample_id in enumerate(sample['id'].tolist()):
                if args.print_vanilla_alignment and args.decoding_path is not None:
                    align_sents[int(sample_id)].append(alignments[int(sample_id)])
       
    if args.decoding_path is not None and args.print_vanilla_alignment:
        with open(os.path.join(args.decoding_path, f'{args.gen_subset}.{args.source_lang}2{args.target_lang}.align'), 'w') as f:
            for sents in align_sents:
                if len(sents)==0:
                    continue                  
                for sent in sents:
                    f.write(str(sent)+'\n')

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == '__main__':
    cli_main()
