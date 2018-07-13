#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import glob
import os
import sys
import random
from datetime import datetime

from collections import defaultdict
from nltk import ngrams

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu, get_logger
from onmt.translate.Translator import make_translator
import onmt.opts

from structuredPredictionNLG.Action import Action

from structuredPredictionNLG.DatasetParser import DatasetParser
from structuredPredictionNLG.DatasetInstance import lexicalize_word_sequence
import numpy
from nltk.translate.bleu_score import corpus_bleu
from copy import copy

from os import listdir
from os.path import isfile, join


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('dataset', type=str, choices=['e2e', 'webnlg', 'sfhotel'])
parser.add_argument('--name', type=str, default='def')
parser.add_argument('--trim', action='store_true', default=False)
parser.add_argument('--full_delex', action='store_true', default=False)
parser.add_argument('--infer_MRs', action='store_true', default=False)

# onmt.opts.py
onmt.opts.add_md_help_argument(parser)
onmt.opts.model_opts(parser)
onmt.opts.train_opts(parser)

opt = parser.parse_args()
opt.reset = False

logger = get_logger(opt.log_file)

if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    logger.info("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    logger.info(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(
        opt.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"),
        comment="Onmt")

progress_step = 0


def report_func(epoch, batch, num_batches,
                progress_step,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        progress_step(int): the progress step.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        msg = report_stats.output(epoch, batch + 1, num_batches, start_time)
        logger.info(msg)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        if opt.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard(
                "progress", writer, lr, progress_step)
        report_stats = onmt.Statistics()

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        # In token batching scheme, the number of sequences is limited
        # such that the total number of src/tgt tokens (including padding)
        # in a batch <= batch_size
        def batch_size_fn(new, count, sofar):
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)

    device = opt.gpuid[0] if opt.gpuid else -1

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def make_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)

    if use_gpu(opt):
        compute.cuda()

    return compute

def get_deletion_precision_cost( i, orig_seq, ref, min=1, max=4):
    del_score = get_deletion_optimal_precision(orig_seq, ref, i, min, max)
    return del_score

def get_deletion_optimal_precision(sequence, ref, i, min=1, max=4):
    seq_after_del = [None, None, None]
    seq_after_del.extend(sequence[:])
    seq_after_del.extend([None, None, None])

    ref_extended = [None, None, None]
    ref_extended.extend(ref[:])
    ref_extended.extend([None, None, None])

    i = i + 3
    fr = i - 2
    if fr < 0:
        fr = 0
    to = i + 3
    if to > len(seq_after_del):
        to = len(seq_after_del)
    seq_after_del = seq_after_del[fr:to]

    rollout_ngram_list = []
    if min <=4 and max >=4:
        rollout_ngram_list.extend(ngrams(seq_after_del, 4, pad_left=False, pad_right=False))
    if min <=3 and max >=3:
        rollout_ngram_list.extend(ngrams(seq_after_del[1:-1], 3, pad_left=False, pad_right=False))
    if min <=2 and max >=2:
        rollout_ngram_list.extend(ngrams(seq_after_del[2:-2], 2, pad_left=False, pad_right=False))
    if min <=1 and max >=1:
        rollout_ngram_list.extend(seq_after_del[3:-3])

    ref_ngram_list = []
    if min <=4 and max >=4:
        ref_ngram_list.extend(ngrams(ref_extended, 4, pad_left=False, pad_right=False))
    if min <=3 and max >=3:
        ref_ngram_list.extend(ngrams(ref_extended[1:-1], 3, pad_left=False, pad_right=False))
    if min <=2 and max >=2:
        ref_ngram_list.extend(ngrams(ref_extended[2:-2], 2, pad_left=False, pad_right=False))
    if min <=1 and max >=1:
        ref_ngram_list.extend(ref_extended[3:-3])

    totalCand = len(rollout_ngram_list)
    if totalCand == 0:
        return False
    for ngram in rollout_ngram_list:
        if ngram not in ref_ngram_list:
            return False
    return True

def infer_post_nonmonotonic_actions(opt_translates):
    non_mono_del_applied_context = {}
    for opt_translate in opt_translates:
        src_lines = []
        non_mono_del_applied_context[opt_translate.predicate] = defaultdict(set)
        with open(opt_translate.src, 'r') as f:
            src_lines = f.readlines()
        with open(opt_translate.output, 'r') as handle:
            lines = [line.strip() for line in handle]
            for i, l in enumerate(lines):
                mr = src_lines[i].strip()
                roll_in = l.split()
                di = opt.parser.test_src_to_di[opt_translate.predicate][src_lines[i].strip()]

                lexicalized_l = lexicalize_word_sequence(l.split(), di.input.delexicalizationMap)
                eval_stats = di.output.evaluateAgainst(lexicalized_l)

                if eval_stats.BLEUSmooth < 1.0:
                    print(mr)
                    print(roll_in)
                    print(eval_stats.BLEU)
                    print(eval_stats.best_ref)

                    index = 0
                    length = len(roll_in)
                    seq_after_del = roll_in[:]

                    while index < length:
                        del_cost = get_deletion_precision_cost(index, seq_after_del, eval_stats.best_ref, min=3)

                        if del_cost:
                            roll_out_action = 0
                            print('------', seq_after_del[index], del_cost)
                            tuple_list = [None, None, None, None]
                            if index - 2 > 0:
                                tuple_list[0] = seq_after_del[index - 2]
                            if index - 1 > 0:
                                tuple_list[1] = seq_after_del[index - 1]
                            if index + 1 < len(seq_after_del):
                                tuple_list[2] = seq_after_del[index + 1]
                            if index + 2 < len(seq_after_del):
                                tuple_list[3] = seq_after_del[index + 2]
                            non_mono_del_applied_context[opt_translate.predicate][seq_after_del[index]].add(tuple(tuple_list))
                            del seq_after_del[index]
                            print('------', seq_after_del)
                            # orig_score = state_sequence[-1].get_word_sequence_optimal_score(seq_after_del, ref_ngram_list)
                            length = length - 1
                        else:
                            roll_out_action = 1
                            index += 1
                        # rnn_labels_del_list.append(roll_out_action)
                    exit()
                    index = 0
                    length = len(seq_after_del)
                    seq_after_ins = seq_after_del[:]
                    seq_after_ins_attrs = seq_after_del_attrs[:]
                    # print()
                    # print()
                    # print(' '.join([o.lower() for o in ref]))
                    # print(' '.join(seq_after_ins))
                    while index < length:
                        ins_cost_vector = state_sequence[-1].get_insertion_precision_cost_vector(index, seq_after_ins,
                                                                                                 ref, min=3)
                        # print(ins_cost_vector)
                        action_probs, action = self.learnedPolicy_ins_descriminator(structuredInstance, seq_after_ins,
                                                                                    seq_after_ins_attrs, index, False,
                                                                                    isTrain=True)
                        rnn_probs_ins_list.append(action_probs)

                        roll_out_action = False
                        for word in ins_cost_vector:
                            if ins_cost_vector[word]:
                                roll_out_action = word
                                break

                        # print(' ' .join(seq_after_ins[:index]))
                        if roll_out_action:
                            # print('++++++', seq_after_ins[index - 1], roll_out_action, seq_after_ins[index])
                            # print(seq_after_ins)
                            tuple_list = [None, seq_after_ins[index], None, None]
                            if index - 1 > 0:
                                tuple_list[0] = seq_after_ins[index - 1]
                            if index + 1 < len(seq_after_ins):
                                tuple_list[2] = seq_after_ins[index + 1]
                            if index + 2 < len(seq_after_ins):
                                tuple_list[3] = seq_after_ins[index + 2]
                            self.non_mono_ins_applied_context[tuple(tuple_list)].add(roll_out_action)
                            seq_after_ins.insert(index, roll_out_action)
                            # print(seq_after_ins)
                            seq_after_ins_attrs.insert(index, seq_after_ins_attrs[index])
                            # orig_score = state_sequence[-1].get_word_sequence_optimal_score(seq_after_ins,ref_ngram_list)
                            length += 1
                        else:
                            index += 1
                        if roll_out_action in self.word2index[structuredInstance.input.predicate]:
                            rnn_labels_ins_list.append(self.word2index[structuredInstance.input.predicate][roll_out_action])
                        else:
                            rnn_labels_ins_list.append(len(self.word2index[structuredInstance.input.predicate]))
                    # print()
                    # print()
                    # print(' '.join(ref))
                    # print(' '.join(orig_seq))
                    # print(' '.join(seq_after_del))
                    # print(' '.join(seq_after_ins))
                    # exit()
                    # insert_cost_vector = state_sequence[-1].get_insertion_cost_vector(True)
                    eval_ceil.append(' '.join(seq_after_ins))

    return corpusBLEU, bleu, rouge, coverage

def check_save_result_path(path):
    save_result_path = os.path.abspath(path)
    model_dirname = os.path.dirname(save_result_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    logger.info('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    logger.info('encoder: ' + str(enc))
    logger.info('decoder: ' + str(dec))


def lazily_load_dataset(corpus_type, opt_per_pred):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt_per_pred.data + opt_per_pred.file_templ + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt_per_pred.data + opt_per_pred.file_templ + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, opt_per_pred, checkpoint):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt_per_pred.train_from)
        fields = onmt.io.load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = onmt.io.load_fields_from_vocab(
            torch.load(opt_per_pred.data + opt_per_pred.file_templ + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        logger.info(' * vocabulary size. target = %d' %
                    (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        logger.info(' * src feature %d size = %d' %
                    (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(' * tgt feature %d size = %d' %
                    (j, len(fields[feat].vocab)))


def build_model(opt, fields, checkpoint):
    logger.info('Building model...')
    model = onmt.ModelConstructor.make_base_model(opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        logger.info('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    logger.info(model)

    return model


def build_optim(model, checkpoint):
    saved_optimizer_state_dict = None

    if opt.train_from:
        logger.info('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        # We need to save a copy of optim.optimizer.state_dict() for setting
        # the, optimizer state later on in Stage 2 in this method, since
        # the method optim.set_parameters(model.parameters()) will overwrite
        # optim.optimizer, and with ith the values stored in
        # optim.optimizer.state_dict()
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        logger.info('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    # Stage 1:
    # Essentially optim.set_parameters (re-)creates and optimizer using
    # model.paramters() as parameters that will be stored in the
    # optim.optimizer.param_groups field of the torch optimizer class.
    # Importantly, this method does not yet load the optimizer state, as
    # essentially it builds a new optimizer with empty optimizer state and
    # parameters from the model.
    optim.set_parameters(model.named_parameters())
    print(
        "Stage 1: Keys after executing optim.set_parameters" +
        "(model.parameters())")
    show_optimizer_state(optim)

    if opt.train_from:
        # Stage 2: In this stage, which is only performed when loading an
        # optimizer from a checkpoint, we load the saved_optimizer_state_dict
        # into the re-created optimizer, to set the optim.optimizer.state
        # field, which was previously empty. For this, we use the optimizer
        # state saved in the "saved_optimizer_state_dict" variable for
        # this purpose.
        # See also: https://github.com/pytorch/pytorch/issues/2830
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        # Convert back the state values to cuda type if applicable
        if use_gpu(opt):
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        print(
            "Stage 2: Keys after executing  optim.optimizer.load_state_dict" +
            "(saved_optimizer_state_dict)")
        show_optimizer_state(optim)

        # We want to make sure that indeed we have a non-empty optimizer state
        # when we loaded an existing model. This should be at least the case
        # for Adam, which saves "exp_avg" and "exp_avg_sq" state
        # (Exponential moving average of gradient and squared gradient values)
        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


# Debugging method for showing the optimizer state
def show_optimizer_state(optim):
    print("optim.optimizer.state_dict()['state'] keys: ")
    for key in optim.optimizer.state_dict()['state'].keys():
        print("optim.optimizer.state_dict()['state'] key: " + str(key))

    print("optim.optimizer.state_dict()['param_groups'] elements: ")
    for element in optim.optimizer.state_dict()['param_groups']:
        print("optim.optimizer.state_dict()['param_groups'] element: " + str(
            element))

def main():
    # load the data!
    if opt.dataset.lower() == 'e2e':
        dataparser = DatasetParser('data/e2e/trainset.csv', 'data/e2e/devset.csv', 'data/e2e/testset_w_refs.csv', 'E2E', opt, light=True)
    elif opt.dataset.lower() == 'webnlg':
        dataparser = DatasetParser('data/webNLG_challenge_data/train', 'data/webNLG_challenge_data/dev', False, 'webNLG', opt, light=True)
    elif opt.dataset.lower() == 'sfhotel':
        dataparser = DatasetParser('data/sfx_data/sfxhotel/train.json', 'data/sfx_data/sfxhotel/valid.json', 'data/sfx_data/sfxhotel/test.json', 'SFHotel', opt, light=True)

    opt.data = 'save_data/{:s}/'.format(opt.dataset)
    gen_templ = dataparser.get_onmt_file_templ(opt)

    opt.parser = dataparser
    model = {}
    fields = {}
    optim = {}
    data_type ={}

    opt_per_pred = {}

    translate_parser = argparse.ArgumentParser(
        description='translate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(translate_parser)
    onmt.opts.translate_opts(translate_parser)
    opt_translate = translate_parser.parse_args(args=[])
    opt_translate.replace_unk = False
    opt_translate.verbose = False
    opt_translate.block_ngram_repeat = False
    if opt.gpuid:
        opt_translate.gpu = opt.gpuid[0]
    opt_translates = []

    dataparser.predicates = ['?request']
    for predicate in dataparser.predicates:
        opt_per_pred[predicate] = copy(opt)
        opt_per_pred[predicate].predicate = predicate
        opt_per_pred[predicate].file_templ = gen_templ.format(predicate)
        opt_per_pred[predicate].save_model = 'save_model/{:s}/{:s}'.format(opt_per_pred[predicate].dataset, predicate)

        # Get the saved model with the highest reported BLEU in dev
        dir_path = 'save_model/{:s}/'.format(opt_per_pred[predicate].dataset)
        poss_models = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.startswith(predicate + "_e") and '_bleuForPred_' in f and '_corpusBLEU_' in f]
        bleu_models = [float(f[f.find('_bleuForPred_') + 13:f.find('_corpusBLEU_')]) for f in poss_models]
        checkpoint_file = 'save_model/{:s}/{:s}'.format(opt_per_pred[predicate].dataset, poss_models[bleu_models.index(max(bleu_models))])

        logger.info('Loading checkpoint from %s' % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        # Peek the fisrt dataset to determine the data_type.
        # (All datasets have the same data_type).
        first_dataset = next(lazily_load_dataset("train", opt_per_pred[predicate]))
        data_type[predicate] = first_dataset.data_type

        # Load fields generated from preprocess phase.
        fields[predicate] = load_fields(first_dataset, data_type[predicate], opt_per_pred[predicate], checkpoint)

        # Report src/tgt features.
        collect_report_features(fields[predicate])

        # Load model.
        model[predicate] = build_model(opt_per_pred[predicate], fields[predicate], checkpoint)
        model[predicate].predicate = predicate
        model[predicate].eval()
        tally_parameters(model[predicate])
        check_save_model_path()

        # Load optimizer.
        optim[predicate] = build_optim(model[predicate], checkpoint)

        opt_translate.predicate = predicate
        opt_translate.batch_size = opt_per_pred[predicate].batch_size
        opt_translate.src = 'cache/train_src_{:s}.txt'.format(opt_per_pred[predicate].file_templ)
        opt_translate.tgt = 'cache/train_eval_refs_{:s}.txt'.format(opt_per_pred[predicate].file_templ)
        opt_translate.output = 'result/{:s}/train_res_{:s}.txt'.format(opt_per_pred[predicate].dataset, opt_per_pred[predicate].file_templ)

        check_save_result_path(opt_translate.output)
        if os.path.isfile(opt_translate.src) and os.path.isfile(opt_translate.tgt):
            translator = make_translator(opt_translate, report_score=False, logger=logger, fields=fields[predicate], model=model[predicate], model_opt=opt_per_pred[predicate])
            translator.output_beam = 'result/{:s}/train_res_beam_{:s}.txt'.format(opt_per_pred[predicate].dataset, opt_per_pred[predicate].file_templ)
            #translator.beam_size = 5
            #translator.n_best = 5
            translator.translate(opt_translate.src_dir, opt_translate.src, opt_translate.tgt, opt_translate.batch_size, opt_translate.attn_debug)
            opt_translates.append(copy(opt_translate))
    infer_post_nonmonotonic_actions(opt_translates)

if __name__ == "__main__":
    main()
