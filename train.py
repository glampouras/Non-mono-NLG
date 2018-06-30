#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import glob
import os
import sys
import random
from datetime import datetime

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

from structuredPredictionNLG.DatasetParser import DatasetParser
from structuredPredictionNLG.DatasetInstance import lexicalize_word_sequence
import numpy
from nltk.translate.bleu_score import corpus_bleu
from copy import copy


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


def train_model(model, fields, optim, data_type, opt_per_pred):
    translate_parser = argparse.ArgumentParser(
        description='translate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(translate_parser)
    onmt.opts.translate_opts(translate_parser)
    opt_translate = translate_parser.parse_args(args=[])
    opt_translate.replace_unk = False
    opt_translate.verbose = False
    opt_translate.block_ngram_repeat = False
    opt_translate.gpu = opt.gpuid[0]

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    trainer = {}
    for predicate in opt.parser.predicates:
        train_loss = make_loss_compute(model[predicate], fields[predicate]["tgt"].vocab, opt_per_pred[predicate])
        valid_loss = make_loss_compute(model[predicate], fields[predicate]["tgt"].vocab, opt_per_pred[predicate],
                                       train=False)
        trainer[predicate] = onmt.Trainer(model[predicate], train_loss, valid_loss, optim[predicate],
                               trunc_size, shard_size, data_type[predicate],
                               norm_method, grad_accum_count)

    logger.info('')
    logger.info('Start training...')
    logger.info(' * number of epochs: %d, starting from Epoch %d' %
                (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    logger.info(' * batch size: %d' % opt.batch_size)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        logger.info('')

        train_stats = {}
        valid_stats = {}
        for predicate in opt.parser.predicates:
            logger.info('Train predicate: %s' % predicate)
            # 1. Train for one epoch on the training set.
            train_iter = make_dataset_iter(lazily_load_dataset("train", opt_per_pred[predicate]),
                                           fields[predicate], opt_per_pred[predicate])
            train_stats[predicate] = trainer[predicate].train(train_iter, epoch, fields[predicate], report_func)
            logger.info('Train perplexity: %g' % train_stats[predicate].ppl())
            logger.info('Train accuracy: %g' % train_stats[predicate].accuracy())

            # 2. Validate on the validation set.
            valid_iter = make_dataset_iter(lazily_load_dataset("valid", opt_per_pred[predicate]),
                                           fields[predicate], opt_per_pred[predicate],
                                           is_train=False)
            valid_stats[predicate] = trainer[predicate].validate(valid_iter)
            logger.info('Validation perplexity: %g' % valid_stats[predicate].ppl())
            logger.info('Validation accuracy: %g' % valid_stats[predicate].accuracy())

            # 3. Log to remote server.
            if opt_per_pred[predicate].exp_host:
                train_stats[predicate].log("train", experiment, optim[predicate].lr)
                valid_stats[predicate].log("valid", experiment, optim[predicate].lr)
            if opt_per_pred[predicate].tensorboard:
                train_stats[predicate].log_tensorboard("train", writer, optim[predicate].lr, epoch)
                train_stats[predicate].log_tensorboard("valid", writer, optim[predicate].lr, epoch)

            # 4. Update the learning rate
            decay = trainer[predicate].epoch_step(valid_stats[predicate].ppl(), epoch)
            if decay:
                logger.info("Decaying learning rate to %g" % trainer[predicate].optim.lr)

        # 5. Drop a checkpoint if needed.
        if epoch % 10 == 0: #epoch >= opt.start_checkpoint_at:
            opt_translates = []
            for predicate in opt.parser.predicates:
                opt_translate.predicate = predicate
                opt_translate.src = 'cache/valid_src_{:s}.txt'.format(opt_per_pred[predicate].file_templ)
                opt_translate.tgt = 'cache/valid_eval_refs_{:s}.txt'.format(opt_per_pred[predicate].file_templ)
                opt_translate.output = 'result/{:s}/valid_res_{:s}.txt'.format(opt_per_pred[predicate].dataset, opt_per_pred[predicate].file_templ)
                #opt_translate.model = '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (
                #opt_per_pred[predicate].save_model, valid_stats[predicate].accuracy(), valid_stats[predicate].ppl(), epoch)

                check_save_result_path(opt_translate.output)

                translator = make_translator(opt_translate, report_score=False, logger=logger, fields=fields[predicate], model=trainer[predicate].model, model_opt=opt_per_pred[predicate])
                translator.output_beam = 'result/{:s}/valid_res_beam_{:s}.txt'.format(opt_per_pred[predicate].dataset, opt_per_pred[predicate].file_templ)
                #translator.beam_size = 5
                #translator.n_best = 5
                translator.translate(opt_translate.src_dir, opt_translate.src, opt_translate.tgt, opt_translate.batch_size, opt_translate.attn_debug)
                opt_translates.append(copy(opt_translate))
            corpusBLEU, bleu, rouge, coverage = evaluate(opt_translates)
            #for predicate in opt.parser.predicates:
            #    trainer[predicate].drop_checkpoint(opt_per_pred[predicate], epoch, corpusBLEU, bleu, rouge, coverage, fields[predicate], valid_stats[predicate])

def evaluate(opt_translates):
    eval_stats = []
    eval_results = []

    for opt_translate in opt_translates:
        with open(opt_translate.output, 'r') as handle:
            lines = [line.strip() for line in handle]
            for i, l in enumerate(lines):
                di = opt.parser.developmentInstances[opt_translate.predicate][i]
                lexicalized_l = lexicalize_word_sequence(l.split(), di.input.delexicalizationMap)

                stats = '\nMR:' + str(di.input.attributeValues) + '\nREAL: ' + ' '.join(lexicalized_l) + '\nDREF: ' + str(di.directReference)
                logger.info(stats)

                eval_stats.append(di.output.evaluateAgainst(lexicalized_l))
                eval_results.append((lexicalized_l, eval_stats[-1].refs))

                stats = '\nEREF: ' + str(eval_stats[-1].refs) + '\nBLEU: ' + str(eval_stats[-1].BLEU) + '\n'
                logger.info(stats)

                if (' '.join(lexicalized_l)).strip() == str(di.directReference).strip() and eval_stats[-1].BLEU != 1.0:
                    exit()

    realizations = []
    references = []
    for realization, refs in eval_results:
        realizations.append(realization)
        references.append(refs)
    corpusBLEU = corpus_bleu(references, realizations)
    bleu = numpy.average([e.BLEU for e in eval_stats])
    rouge = numpy.average([e.ROUGE for e in eval_stats])
    coverage = numpy.average([e.COVERAGE for e in eval_stats])

    print("corpusBLEU:", corpusBLEU)
    print("BLEU:", bleu)
    print("smoothBLEU:", numpy.average([e.BLEUSmooth for e in eval_stats]))
    print("ROUGE:", rouge)
    print("COVERAGE:", coverage)

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
    # load the training data!
    if opt.dataset.lower() == 'e2e':
        dataparser = DatasetParser('data/e2e/trainset.csv', 'data/e2e/devset.csv', 'data/e2e/testset_w_refs.csv', 'E2E', opt)
    elif opt.dataset.lower() == 'webnlg':
        dataparser = DatasetParser('data/webNLG_challenge_data/train', 'data/webNLG_challenge_data/dev', False, 'webNLG', opt)
    elif opt.dataset.lower() == 'sfhotel':
        dataparser = DatasetParser('data/sfx_data/sfxhotel/train.json', 'data/sfx_data/sfxhotel/valid.json', 'data/sfx_data/sfxhotel/test.json', 'SFHotel', opt)

    opt.data = 'save_data/{:s}/'.format(opt.dataset)
    gen_templ = dataparser.get_onmt_file_templ(opt)

    dataparser.predicates = ['inform']
    opt.parser = dataparser
    model = {}
    fields = {}
    optim = {}
    data_type ={}

    opt_per_pred = {}
    for predicate in dataparser.predicates:
        opt_per_pred[predicate] = copy(opt)
        opt_per_pred[predicate].predicate = predicate
        opt_per_pred[predicate].file_templ = gen_templ.format(predicate)
        opt_per_pred[predicate].save_model = 'save_model/{:s}/{:s}'.format(opt_per_pred[predicate].dataset, predicate)
        # Load checkpoint if we resume from a previous training.
        if opt_per_pred[predicate].train_from:
            logger.info('Loading checkpoint from %s' % opt_per_pred[predicate].train_from)
            checkpoint = torch.load(opt_per_pred[predicate].train_from,
                                    map_location=lambda storage, loc: storage)
            opt_per_pred[predicate] = checkpoint['opt']
            # I don't like reassigning attributes of opt: it's not clear.
            opt_per_pred[predicate].start_epoch = checkpoint['epoch'] + 1
            opt_per_pred[predicate].save_model = opt_per_pred[predicate].save_model
        else:
            checkpoint = None

        # Peek the fisrt dataset to determine the data_type.
        # (All datasets have the same data_type).
        first_dataset = next(lazily_load_dataset("train", opt_per_pred[predicate]))
        data_type[predicate] = first_dataset.data_type

        # Load fields generated from preprocess phase.
        fields[predicate] = load_fields(first_dataset, data_type[predicate], opt_per_pred[predicate], checkpoint)

        # Report src/tgt features.
        collect_report_features(fields[predicate])

        # Build model.
        model[predicate] = build_model(opt_per_pred[predicate], fields[predicate], checkpoint)
        model[predicate].predicate = predicate
        tally_parameters(model[predicate])
        check_save_model_path()

        # Build optimizer.
        optim[predicate] = build_optim(model[predicate], checkpoint)

    # Do training.
    train_model(model, fields, optim, data_type, opt_per_pred)

    # Do

    # If using tensorboard for logging, close the writer after training.
    if opt.tensorboard:
        writer.close()

        # Do training.
        train_model(model, fields, optim, data_type, model_opt)

        # If using tensorboard for logging, close the writer after training.
        if opt.tensorboard:
            writer.close()


if __name__ == "__main__":
    main()
