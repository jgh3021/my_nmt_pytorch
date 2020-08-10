from __future__ import print_function
import argparse
import json
import sys

import torch
from torch.nn import functional as F

def evaluate(model, val_iter, vocab_size, DE, EN):
    with torch.no_grad():
        model.eval()
        pad = EN.vocab.stoi['<pad>']
        total_loss = 0
        for b, batch in enumerate(val_iter):
            src, len_src = batch.src
            trg, len_trg = batch.trg
            src = src.data.cuda()
            trg = trg.data.cuda()
            output = model(src, trg, teacher_forcing_ratio=0.0)
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                   trg[1:].contiguous().view(-1),
                                   ignore_index=pad)
            total_loss += loss.data.item()
        return total_loss / len(val_iter)

if __name__ == '__main__':
    expected_version = 'WMT_v1.0'
    parser = argparse.ArgumentParser(
        description='Evaluation for KorQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        read_version = "_".join(dataset_json['version'].split("_")[:-1])
        if (read_version != expected_version):
            print('Evaluation expects ' + expected_version +
                  ', but got dataset with ' + read_version,
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))