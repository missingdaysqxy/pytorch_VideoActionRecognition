# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 17:07
# @Author  : liuqixuan_i
# @Email   : liuqixuan_i@didiglobal.com
# @File    : dataset.py
# @Software: PyCharm

import os
import csv
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from core import Config


class ActionDataset(Dataset):
    def __init__(self, path: str, class_dict: dict):
        assert os.path.isfile(path), "file %s not exist" % path
        self.classes = list(class_dict.keys())
        self.data = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append([class_dict[row[0]], row[1]])

    def __getitem__(self, idx):
        label, path = self.data[idx]
        print(label, path)
        cap = cv2.VideoCapture(path)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        data = np.zeros((h, w, 3 * fc))
        for i in range(fc):
            ret, frame = cap.read()
            cv2.imshow(label, frame)
            cv2.waitKey(5)
            data[i * 3:(i + 1) * 3] = frame
        return label, data

    def __len__(self):
        return len(self.data)


def ActionDataloader(data_type, config: Config):
    # type:(str,Config)->DataLoader
    assert data_type in ["train", "training", "val", "validation", "inference"]
    if data_type in ["train", "training"]:
        data_path = config.train_data_path
        shuffle = config.shuffle_train
        drop_last = config.drop_last_train
    elif data_type in ["val", "validation", "inference"]:
        data_path = config.val_data_path
        shuffle = config.shuffle_val
        drop_last = config.drop_last_val
    dataset = ActionDataset(data_path, config.classes)
    assert len(dataset) > config.batch_size
    return DataLoader(dataset, config.batch_size, shuffle, num_workers=config.num_data_workers,
                      pin_memory=config.pin_memory, drop_last=drop_last, timeout=config.time_out)


if __name__ == "__main__":
    config = Config("train")
    ds = ActionDataset(config.train_data_path, config.classes)
    print(ds[0])
