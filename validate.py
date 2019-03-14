# -*- coding: utf-8 -*-
# @Time    : 2019/1/9 21:19
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : validate.py
# @Software: PyCharm

import os
import math
import warnings
import numpy as np
import torch as t
from torch.nn import Module
from torchnet import meter
from collections import defaultdict
from core import *


def get_data(data_type, config: Config):
    data = ActionDataLoader(data_type, config)
    return data


def validate(model, val_data, config, vis):
    # type: (Module,ActionDataLoader,Config,Visualizer)->(float, np.ndarray, defaultdict)
    with t.no_grad():
        correct_label_distrib = defaultdict(int)  # show how many sub-images are correct for all parent-images
        confusion_matrix = meter.ConfusionMeter(config.num_classes)
        # validate
        for i, b_data in enumerate(val_data):
            model.eval()
            # input data
            b_labels, b_actions = b_data
            if config.use_gpu:
                with t.cuda.device(0):
                    b_labels = b_labels.cuda()
                    b_actions = b_actions.cuda()
            # b_labels = b_labels.view(-1)
            # forward
            b_probs = model(b_actions)
            # confusion matrix statistic
            b_preds = t.argmax(b_probs, dim=-1)
            confusion_matrix.add(b_preds, b_labels)
            # distribution statistic
            equals = t.split((b_preds == b_labels), 8)
            for eqs in equals:
                sub_corr_count = int(eqs.sum().cpu().numpy())
                correct_label_distrib[sub_corr_count] += 1
            for i in range(9):
                correct_label_distrib[i] += 0
            correct_label_distrib = dict(sorted(correct_label_distrib.items(), key=lambda x: x[0], reverse=False))
            # print process
            if i % config.ckpt_freq == 0 or i >= len(val_data) - 1:
                val_cm = confusion_matrix.value()
                msg = "[Validation]process:{}/{},".format(i, len(val_data) - 1)
                msg += "confusion matrix:\n{}\ncorrect labels distribution:\n{}\n".format(
                    val_cm, correct_label_distrib)
                vis.log_process(i, len(val_data) - 1, msg, 'val_log', append=True)

                vis.bar(list(correct_label_distrib.values()), 'number of correct labels',
                        list(correct_label_distrib.keys()))

        val_cm = confusion_matrix.value()
        precision = val_cm[0, 0] / val_cm[0].sum()
        recall = val_cm[0, 0] / val_cm[:, 0].sum()
        val_f1 = 2 * precision * recall / (precision + recall)
        # val_acc = val_cm.trace().astype(np.float) / val_cm.sum()

    return val_f1, val_cm, correct_label_distrib


def main(args):
    config = Config('inference')
    print(config)
    val_data = get_data("val", config)
    model = get_model(config)
    vis = Visualizer(config)
    print("Prepare to validate model...")

    val_f1, val_cm, corr_label = validate(model, val_data, config, vis)
    msg = 'validation f1-score:{}\n'.format(val_f1)
    msg += 'validation confusion matrix:\n{}\n'.format(val_cm)
    msg += 'number of correct labels in a scene:\n{}\n'.format(corr_label)
    print("Validation Finish!", msg)
    vis.log(msg, 'val_result', log_file=config.val_result)
    print("save best validation result into " + config.val_result)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()

    main(args)
