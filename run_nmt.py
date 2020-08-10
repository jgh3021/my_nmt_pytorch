from __future__ import absolute_import, division, print_function

import re
import argparse
import logging
import os
import random
import math
import sys

import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset
from tqdm import tqdm, trange

from models.modeling_bert import MachineTranslation, Config
from utils.optimization import AdamW, WarmupLinearSchedule
from utils.tokenization import BertTokenizer
from utils.nmt_utils import read_nmt_examples, convert_examples_to_features

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train(e, model, optimizer, train_iter, vocab_size, grad_clip, DE, EN):
    model.train()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0




def load_and_cache_examples(batch_size):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint", default='pretrain_ckpt/large_v1_model.bin', type=str,
                        help="pre-trained model checkpoint")
    parser.add_argument("--config_file", default='data/large_config.json', type=str,
                        help="model configuration file")
    parser.add_argument("--vocab_file", default='data/large_v1_32k_vocab.txt', type=str,
                        help="tokenizer vocab file")
    parser.add_argument("--train_file", default='data/korquad/KorQuAD_v1.0_train.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--seed', default=42, type=int,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', default='O2', type=str,
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="For distributed training: local_rank")
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    tokenizer = BertTokenizer(args.vocab_file, max_len=args.max_seq_length, do_basic_tokenize=True)

    # Prepare model
    config = Config.from_json_file(args.config_file)
    model = (config)
    model.bert.load_state_dict(torch.load(args.checkpoint))
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    model.to(device)

    logger.info("Training hyper-parameters %s", args)

    # Training
    train_dataset = load_and_cache_examples(args, tokenizer)
    train(args, train_dataset, model)
    #############korean nlp ############################################

    #######################seq2seq########################
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter,
              en_size, args.grad_clip, DE, EN)
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)