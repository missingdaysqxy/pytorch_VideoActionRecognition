# -*- coding: utf-8 -*-
# @Time    : 2019/1/13/013 20:00 下午
# @Author  : qixuan
# @Email   : qixuan.lqx@qq.com
# @File    : _base.py
# @Software: PyCharm

import os
import time
import torch as t
from typing import Dict
from torch import save, load, set_grad_enabled
from warnings import warn
from torch.nn import Module, DataParallel
from torch.optim import Optimizer
from .config import Config


class _BaseModule(Module):
    def __int__(self, config: Config):
        super(_BaseModule, self).__init__()
        self.config = config

    def forward(self, input):
        raise NotImplementedError("Should be overridden by all subclasses.")

    def initialize_weights(self):
        raise NotImplementedError("Should be overridden by all subclasses.")


def get_model(config: Config, **kwargs) -> _BaseModule:
    """
    Find a Module specified by config.module from core.models, and get an instance of it
    :param config: instance of core.Config
    :param state_dict_preprocess: a function to do some pre-work with state_dict before it will be loaded
    :param kwargs: arguments that will be passed into the Module, default is None
    :return: an instance of the Module specified by config.module
    """
    assert isinstance(config, Config)
    from . import models
    try:
        with set_grad_enabled(config.enable_grad):
            model = getattr(models, config.module)(config, **kwargs)
    except AttributeError as e:
        raise AttributeError(
            "No module named '{}' exists in core.models, error message: {}".format(config.module, e))
    from warnings import warn
    # move core to GPU
    if config.use_gpu:
        model = model.cuda(0)
    # initialize weights
    try:
        getattr(model, "initialize_weights")()
    except AttributeError as e:
        warn("initialize weights failed because:\n" + str(e))
    # parallel processing
    model = DataParallel(model, config.gpu_list)
    # load weights
    if os.path.exists(config.weight_load_path):
        try:
            state_dict = load(config.weight_load_path, map_location=config.map_location)
            model.load_state_dict(state_dict)
            print('Loaded weights from ' + config.weight_load_path)
        except RuntimeError as e:
            warn("Failed to load weights file {} because:\n{}, try to auto-fit...".format(config.weight_load_path, e))
            if config.weight_autofit:
                autofit_weights(state_dict, model)
            else:
                raise e
    return model


# def preprocess_state_dict(state_dict, model:Module):
#     for k, v in state_dict.items():
#         print("{:30} shape:{}".format(k, tuple(v.shape)))
#
#     for k, v in model.state_dict().items():
#         print("{:30} shape:{}".format(k, tuple(v.shape)))

def autofit_weights(state_dict:dict, model:Module):
    def cross_modality_pretrain(layer_weight, target_shape):
        # transform the original 3 channel weight to "channel" channel
        S = 0
        for i in range(3):
            S += layer_weight[:, i, :, :]
        avg = S / 3.
        new_layer_weight = t.zeros(target_shape,dtype=layer_weight.dtype)
        # todo:先将shape的长度统一
        # todo:再将shape的维度平均
        for i in range(channel):
            new_layer_weight[:, i, :, :] = avg.data
        return new_layer_weight

    model_dict = model.state_dict()
    for k,v in model_dict.items():
        if k in state_dict:
            if v.shape != state_dict[k].shape:
                state_dict[k]=cross_modality_pretrain(state_dict[k],v.shape)

def make_checkpoint(config, epoch, start_time, loss_val, train_acc, val_acc, model, optimizer=None):
    # type:(Config,int,float,float,float,float,Module,Optimizer)->None
    """
    generate temporary training process data for resuming by resume_checkpoint()
    """
    save(model.state_dict(), config.temp_weight_path)
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        save(optimizer.state_dict(), config.temp_optim_path)
    with open(config.train_record_file, 'a+') as f:
        elapsed_time = time.time() - start_time,
        record = config.__record_dict__.format(config.init_time, epoch, start_time, elapsed_time, loss_val, train_acc,
                                               val_acc)
        f.write(record + '\n')


def resume_checkpoint(config: Config, model: Module, optimizer: Optimizer = None) -> int:
    """
    resume training process data from config.logs which generated by make_checkpoint()
    :return number of last epoch
    """
    last_epoch = -1
    temp_weight_path = config.temp_weight_path
    temp_optim_path = config.temp_optim_path
    if os.path.exists(config.train_record_file):
        try:
            with open(config.train_record_file, 'r') as f:
                last = f.readlines()[-1]
                import json
                info = json.loads(last)
                last_epoch = int(info["epoch"])
                last_init = str(info["init"])
                if not os.path.exists(temp_weight_path):
                    temp_weight_path = temp_weight_path.replace(config.init_time, last_init)
                if not os.path.exists(temp_optim_path):
                    temp_optim_path = temp_optim_path.replace(config.init_time, last_init)
            print("Continue train from last epoch %d" % last_epoch)
        except:
            warn("Rename invalid train record file from {} to {}".format(config.train_record_file,
                                                                         config.train_record_file + '.badfile'))
            warn("Can't get last_epoch value, {} will be returned".format(last_epoch))
            os.rename(config.train_record_file, config.train_record_file + '.badfile')
    if os.path.exists(temp_weight_path):
        try:
            model.load_state_dict(load(temp_weight_path))
            print("Resumed weight checkpoint from {}".format(temp_weight_path))
        except:
            warn("Move invalid temp {} weights file from {} to {}".format(type(model), temp_weight_path,
                                                                          temp_weight_path + '.badfile'))
            os.rename(temp_weight_path, temp_weight_path + '.badfile')
    if optimizer is not None and os.path.exists(temp_optim_path):
        try:
            optimizer.load_state_dict(load(temp_optim_path))
            print("Resumed optimizer checkpoint from {}".format(temp_optim_path))
        except:
            warn("Move invalid temp {} weights file from {} to {}".format(type(optimizer), temp_optim_path,
                                                                          temp_optim_path + '.badfile'))
            os.rename(temp_optim_path, temp_optim_path + '.badfile')

    return last_epoch
