# -*- coding: utf-8 -*-
# @Time    : 2019/1/13/013 20:00 下午
# @Author  : qixuan
# @Email   : qixuan.lqx@qq.com
# @File    : _base.py
# @Software: PyCharm

import os
import time
from torch import save, load, set_grad_enabled
from warnings import warn
from torch.nn import Module, DataParallel
from torch.optim import Optimizer
from core.config import Config


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
    :param kwargs: arguments that will be passed into the Module, default is None
    :return: an instance of the Module specified by config.module
    """
    assert isinstance(config, Config)
    from core import models
    try:
        with set_grad_enabled(config.enable_grad):
            model = getattr(models, config.module)(config, **kwargs)
    except AttributeError as e:
        import sys
        raise AttributeError(
            "No module named '{}' exists in core.models, error message: {}".format(config.module,e))
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
            warn("Failed to load weights file")
            print('Failed to load weights file {} because:\n{}'.format(config.weight_load_path, e))
    return model


def make_checkpoint(config, epoch, start_time, loss_val, train_acc, val_acc, model, optimizer=None):
    # type:(Config,int,str,str,float,float,float,Module,Optimizer)->None
    """
    generate temporary training process data for resuming by resume_checkpoint()
    """
    save(model.state_dict(), config.temp_weight_path)
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        save(optimizer.state_dict(), config.temp_optim_path)
    with open(config.train_record_file, 'a+') as f:
        elapsed_time = time.time() - start_time,
        record = config.__record_dict__.format(epoch, start_time, elapsed_time, loss_val, train_acc, val_acc)
        f.write(record + '\n')


def resume_checkpoint(config: Config, model: Module, optimizer: Optimizer = None) -> int:
    """
    resume training process data from config.logs which generated by make_checkpoint()
    :return number of last epoch
    """
    last_epoch = -1
    if os.path.exists(config.temp_weight_path):
        try:
            model.load_state_dict(load(config.temp_weight_path))
            print("Resumed weight checkpoint from {}".format(config.temp_weight_path))
        except:
            warn("Move invalid temp {} weights file from {} to {}".format(type(model), config.temp_weight_path,
                                                                          config.temp_weight_path + '.badfile'))
            os.rename(config.temp_weight_path, config.temp_weight_path + '.badfile')
    if optimizer is not None and os.path.exists(config.temp_optim_path):
        try:
            optimizer.load_state_dict(load(config.temp_optim_path))
            print("Resumed optimizer checkpoint from {}".format(config.temp_optim_path))
        except:
            warn("Move invalid temp {} weights file from {} to {}".format(type(optimizer), config.temp_optim_path,
                                                                          config.temp_optim_path + '.badfile'))
            os.rename(config.temp_optim_path, config.temp_optim_path + '.badfile')
    if os.path.exists(config.train_record_file):
        try:
            with open(config.train_record_file, 'r') as f:
                last = f.readlines()[-1]
                import json
                info = json.loads(last)
                last_epoch = int(info["epoch"])
            print("Continue train from last epoch %d" % last_epoch)
        except:
            warn("Move invalid train record file from {} to {}".format(config.train_record_file,
                                                                       config.train_record_file + '.badfile'))
            warn("Can't get last_epoch value, {} will be returned".format(last_epoch))
            os.rename(config.train_record_file, config.train_record_file + '.badfile')
    return last_epoch


if __name__ == "__main__":
    pass
