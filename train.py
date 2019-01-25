# coding=UTF-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from pyrouge import Rouge155
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from data_utils import Data
from generator import Generator
from model import RewardWeightedCrossEntropyLoss
from my_args import args


def calc_rouge_score(system_dir, gold_dir):
    r = Rouge155(args.rouge_path)
    r.system_dir = system_dir
    r.model_dir = gold_dir
    r.system_filename_pattern = '([a-zA-Z0-9]*_?[0-9]?).summary'
    r.model_filename_pattern = '([A-Z]\.)?#ID#.gold'
    output = r.convert_and_evaluate(rouge_args='-e %s -a -c 95 -m -n 4 -w 1.2' % args.rouge_data_directory)
    output_dict = r.output_to_dict(output)

    return output_dict["rouge_1_f_score"], output_dict["rouge_2_f_score"], output_dict["rouge_l_f_score"]

def train():
    # Prepare data
    print("Prepare training data...")
    train_data= Data("training")
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    print("Prepare validation data...")
    validation_data = Data("validation")
    validation_data_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)

    # Build model
    print("Build model...")
    model = Generator()
    if torch.cuda.is_available() and args.use_gpu:
        model.cuda()

    loss_fn = RewardWeightedCrossEntropyLoss() if args.rl else nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.learning_rate_G*10, momentum=0.9)

    writer = SummaryWriter(args.train_dir)

    # Start training
    for epoch in range(args.train_epoches):
        print("Epoch %d: Start training" % epoch)
        train_losses = 0
        val_losses = 0
        for i_batch, sampled_batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            _, doc, label, mlabel, rewards, weights, _ = sampled_batch
            probs, logits = model(doc)
            if args.rl:
                loss = loss_fn(probs.contiguous().view(-1, args.target_label_size), mlabel.view(-1), rewards.view(-1), weights.view(-1))
            else:
                loss = loss_fn(probs.contiguous().view(-1, args.target_label_size), label.view(-1))
            train_losses += loss.data * probs.size(0)
            print(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        train_losses /= len(train_data)

        # Save model
        print("Epoch %d: Saving model" % epoch)
        torch.save(model.state_dict(), os.path.join(args.train_dir, "epoch-%d.model" % epoch))

        # Validate model
        print("Epoch %d: Validate model performance" % epoch)
        for i_batch, sampled_batch in enumerate(validation_data_loader):
            docname, doc, label, mlabel, rewards, weights, _ = sampled_batch
            probs, logits = model(doc)
            if args.rl:
                val_losses += loss_fn(probs.contiguous().view(-1, args.target_label_size), mlabel.view(-1), rewards.view(-1), weights.view(-1)).data * probs.size(0)
            else:
                val_losses += loss_fn(probs.contiguous().view(-1, args.target_label_size), label.view(-1)).data * probs.size(0)
            validation_data.write_prediction_summary(docname, logits, "epoch-%d.model" % epoch)
        val_losses /= len(validation_data)

        # ROUGE Evaluation
        system_dir = os.path.join(args.train_dir, '.'.join(["epoch-%d.model" % epoch, "validation", "summary"]))
        model_dir = os.path.join(args.gold_summary_directory, "gold-%s-validation-orgcase" % args.data_mode)
        r1, r2, rl = calc_rouge_score(system_dir, model_dir)
        print("Epoch %d: ROUGE Score(1, 2, L) %.2f %.2f %.2f"% (epoch, r1*100, r2*100, rl*100))

        # Write to Summary
        writer.add_scalar('training_loss', train_losses, epoch)
        writer.add_scalar('validation_loss', val_losses, epoch)
        writer.add_scalars('rouge_score', {'rouge-1': r1,
                                           'rouge-2': r2,
                                           'rouge-L': rl}, epoch)

    writer.export_scalars_to_json('./%s/all_scalars.json' % args.train_dir)
    writer.close()

if __name__ == '__main__':
    train()
    """for epoch in range(20):
        system_dir = os.path.join(args.train_dir, '.'.join(["epoch-%d.model" % epoch, "validation", "summary"]))
        model_dir = os.path.join(args.gold_summary_directory, "gold-%s-validation-orgcase" % args.data_mode)
        r1, r2, rl = calc_rouge_score(system_dir, model_dir)
        print(r1, r2, rl)"""
