# coding=UTF-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from pyrouge import Rouge155
from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from data_utils import Data
from generator import Generator
from model import RewardWeightedCrossEntropyLoss
from my_args import args

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    """cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")"""
    cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

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
    #optimizer = SGD(model.parameters(), lr=args.learning_rate_G*10, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=args.learning_rate_G)

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

def test():
    # Prepare data
    print("Prepare test data...")
    test_data= Data("test")
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # Build model
    print("Build model...")
    model = Generator()
    if torch.cuda.is_available() and args.use_gpu:
        model.cuda()

    print("Load model...")
    model.load_state_dict(torch.load(os.path.join(args.train_dir, "epoch-%d.model" % args.model_to_load)))
    model.eval()

    loss_fn = RewardWeightedCrossEntropyLoss() if args.rl else nn.CrossEntropyLoss()

    #writer = SummaryWriter(args.train_dir)

    # Start training
    test_losses = 0
    # Test model
    print("Test model performance")
    for i_batch, sampled_batch in enumerate(test_data_loader):
        docname, doc, label, mlabel, rewards, weights, _ = sampled_batch
        probs, logits = model(doc)
        if args.rl:
            test_losses += loss_fn(probs.contiguous().view(-1, args.target_label_size), mlabel.view(-1), rewards.view(-1), weights.view(-1)).data * probs.size(0)
        else:
            test_losses += loss_fn(probs.contiguous().view(-1, args.target_label_size), label.view(-1)).data * probs.size(0)
        if args.attn and args.coverage:
            test_losses -= torch.sum(torch.mean(model.covloss, dim=1)).squeeze()
        test_data.write_prediction_summary(docname, logits, "epoch-%d.model" % args.model_to_load)
    test_losses /= len(test_data)

    # ROUGE Evaluation
    system_dir = os.path.join(args.train_dir, '.'.join(["epoch-%d.model" % args.model_to_load, "test", "summary"]))
    model_dir = os.path.join(args.gold_summary_directory, "gold-%s-test-orgcase" % args.data_mode)
    r1, r2, rl = calc_rouge_score(system_dir, model_dir)
    print("ROUGE Score(1, 2, L) %.2f %.2f %.2f"% (r1*100, r2*100, rl*100))

    # Draw attention weight
    for i in range(len(test_data)):
        attn_weight = model.all_attn_weights[i].detach()#.numpy()
        label, weight = mlabel[i], weights[i]
        logit = F.softmax(probs, dim=2)[i][:, 1].squeeze().detach().numpy()

        result = torch.masked_select(attn_weight, weight.eq(1)).view(args.max_doc_length, -1).transpose(0, 1)
        length = result.size(0)
        result = torch.masked_select(result, weight.eq(1)).view(length, length).transpose(0, 1).numpy()

        fig, ax = plt.subplots()
        im, cbar = heatmap(result, list(range(1, length+1)), list(range(1, length+1)), ax=ax,
                    cmap="YlGn")

        # We want to show all ticks...
        ax.set_xticks(np.arange(length))
        ax.set_yticks(np.arange(length))
        # ... and label them with the respective list entries
        ax.set_xticklabels(list(range(1, length+1)))
        ax.set_yticklabels(["%d(%.2f)" % (idx,l) for idx, l in zip(range(1, length+1), logit)])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                rotation_mode="anchor")

        fig.tight_layout()
        plt.savefig(os.path.join(args.train_dir, "Figure_%d" % i))

    """# Write to Summary
    writer.add_scalar('training_loss', train_losses, epoch)
    writer.add_scalar('validation_loss', val_losses, epoch)
    writer.add_scalars('rouge_score', {'rouge-1': r1,
                                        'rouge-2': r2,
                                        'rouge-L': rl}, epoch)

    writer.export_scalars_to_json('./%s/all_scalars.json' % args.train_dir)
    writer.close()"""

if __name__ == '__main__':
    print(args)
    (args.exp_mode == "train" and train()) or test()

    """for epoch in range(20):
        system_dir = os.path.join(args.train_dir, '.'.join(["epoch-%d.model" % epoch, "validation", "summary"]))
        model_dir = os.path.join(args.gold_summary_directory, "gold-%s-validation-orgcase" % args.data_mode)
        r1, r2, rl = calc_rouge_score(system_dir, model_dir)
        print(r1, r2, rl)"""
