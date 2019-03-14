# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 13:30
# @Author  : liuqixuan_i
# @Email   : liuqixuan_i@didiglobal.com
# @File    : train.py
# @Software: PyCharm


import os
import time
import numpy as np
import torch as t
from warnings import warn
from collections import defaultdict
from torch.nn import Module
from torchnet import meter
from core import *
from validate import validate as val
import ipdb


def get_data(data_type, config: Config):
    data = ActionDataLoader(data_type, config)
    return data


def get_loss_functions(config: Config) -> Module:
    if config.loss_type == "mse":
        return t.nn.MSELoss()
    elif config.loss_type in ["cross_entropy", "crossentropy", "cross", "ce"]:
        return t.nn.CrossEntropyLoss()
    else:
        raise RuntimeError("Invalid config.loss_type:" + config.loss_type)


def get_optimizer(model: Module, config: Config) -> t.optim.Optimizer:
    if config.optimizer == "sgd":
        return t.optim.SGD(model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        return t.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError("Invalid value: config.optimizer")


def train(model, train_data, val_data, config, vis):
    # type: (Module,ActionDataLoader,ActionDataLoader,Config,Visualizer)->None

    # init loss and optim
    criterion = get_loss_functions(config)
    optimizer = get_optimizer(model, config)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, 1, config.lr_decay)
    # try to resume
    last_epoch = resume_checkpoint(config, model, optimizer)
    assert last_epoch + 1 < config.max_epoch, \
        "previous training has reached epoch {}, please increase the max_epoch in {}". \
            format(last_epoch + 1, type(config))
    if last_epoch == -1:  # start a new train proc
        vis.save(config.vis_env_path + 'last')
        vis.clear()
    # init meter statistics
    loss_meter = meter.AverageValueMeter()
    AP_meter = meter.APMeter()
    # recall_meter=meter.RecallMeter()
    confusion_matrix = meter.ConfusionMeter(config.num_classes)
    last_score = 0
    for epoch in range(last_epoch + 1, config.max_epoch):
        epoch_start = time.time()
        loss_mean = None
        train_acc = 0
        scheduler.step(epoch)
        loss_meter.reset()
        AP_meter.reset()
        confusion_matrix.reset()
        model.train()
        for i, b_data in enumerate(train_data):
            # input data
            b_labels, b_actions = b_data
            if config.use_gpu:
                with t.cuda.device(0):
                    b_labels = b_labels.cuda()
                    b_actions = b_actions.cuda()
                    criterion = criterion.cuda()
            b_actions.requires_grad_(True)
            b_labels.requires_grad_(False)
            # b_labels = b_labels.view(-1)
            # forward
            # ipdb.set_trace()
            b_probs = model(b_actions)
            loss = criterion(b_probs, b_labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistic
            loss_meter.add(loss.data.cpu())
            b_preds = t.argmax(b_probs, dim=-1)
            AP_meter.add(b_preds, b_labels)
            confusion_matrix.add(b_preds, b_labels)
            # print process
            if i % config.ckpt_freq == 0 or i >= len(train_data) - 1:
                step = epoch * len(train_data) + i
                loss_mean = loss_meter.value()[0]
                cm_value = confusion_matrix.value()
                num_correct = cm_value.trace().astype(np.float)
                train_acc = num_correct / cm_value.sum()
                AP = AP_meter.value()
                cm_AP = cm_value[0, 0] / cm_value[0].sum()
                cm_recall = cm_value[0, 0] / cm_value[:, 0].sum()
                f1_score = 2 * cm_AP * cm_recall / (cm_AP + cm_recall)
                vis.plot(loss_mean, step, 'Loss_Value', "Loss Curve")
                vis.plot(train_acc, step, 'train_acc', 'Training Accuracy')
                vis.plot(f1_score, step, 'train_F1', 'Training F1 Score')
                lr = optimizer.param_groups[0]['lr']
                msg = "epoch:{},iteration:{}/{},loss:{},AP:{},cmAP:{},recall:{},lr:{},confusion_matrix:\n{}".format(
                    epoch, i, len(train_data) - 1, loss_mean, AP, cm_AP, cm_recall, lr, confusion_matrix.value())
                vis.log_process(i, len(train_data) - 1, msg, 'train_log')

                # check if debug file occur
                if os.path.exists(config.debug_flag_file):
                    ipdb.set_trace()
        # validate after each epoch
        val_f1, val_cm, corr_label = val(model, val_data, config, vis)
        vis.plot(val_f1, epoch, 'val_f1', 'Validation F1 Score', ['val_f1'])
        # save checkpoint
        if val_f1 > last_score:
            msg = 'Best validation result after epoch {}, loss:{}, train_acc: {}'.format(epoch, loss_mean, train_acc)
            msg += 'validation f1-score:{}\n'.format(val_f1)
            msg += 'validation confusion matrix:\n{}\n'.format(val_cm)
            msg += 'number of correct labels in a scene:\n{}\n'.format(corr_label)
            vis.log(msg, 'best_val_result', log_file=config.val_result, append=False)
            print("save best validation result into " + config.val_result)
        last_score = val_f1
        make_checkpoint(config, epoch, epoch_start, loss_mean, train_acc, val_f1, model, optimizer)


def main(*args, **kwargs):
    config = Config('train', **kwargs)
    print(config)
    train_data = get_data("train", config)
    val_data = get_data("val", config)
    model = get_model(config)
    vis = Visualizer(config)
    print("Prepare to train model...")
    train(model, train_data, val_data, config, vis)
    # save core
    print("Training Finish! Saving model...")
    try:
        t.save(model.state_dict(), config.weight_save_path)
        os.remove(config.temp_optim_path)
        os.remove(config.temp_weight_path)
        print("Model saved into " + config.weight_save_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to save model because {}, check temp weight file in {}".format(e, config.temp_weight_path))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()

    main(args)
