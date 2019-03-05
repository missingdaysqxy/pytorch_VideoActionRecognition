# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 17:15
# @Author  : liuqixuan_i
# @Email   : liuqixuan_i@didiglobal.com
# @File    : prepare_data.py
# @Software: PyCharm

import os
import re
import cv2
import csv
import h5py
from warnings import warn
from pose import decode_pose, align_skeletons
from multiprocessing import Pool
from functools import partial

PROC_EXT = ".proch5.mp4"
assert os.path.splitext(PROC_EXT)[1] in [".mp4"], "must save as .mp4 video file!"

_REGEX_ = re.compile("^(?P<seq>\d+)_(?P<label>\w+)_(?P<num>\d+){}$".format(PROC_EXT.replace(".", r"\.")))


def _get_label(cls, file_name):
    if cls == "neg":
        return "negative"
    else:
        reg = _REGEX_
        rmtch = reg.match(file_name)
        if rmtch:
            return rmtch.groupdict()["label"]
        else:
            return "unknown_positive"


def save_skeletons2video(person_skeletons, save_path, show_window=False):
    assert len(person_skeletons) > 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 20, person_skeletons[0].shape[1::-1])
    for frm in person_skeletons:
        out.write(frm)
        if show_window:
            cv2.imshow(os.path.basename(proc_path), frm)
            if cv2.waitKey(1) == ord('q'):
                break
                cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    out.release()


def proc_h5(h5_path, zoom_factor=1.0, show_window=True, rebuild=False):
    save_path = os.path.splitext(os.path.splitext(h5_path)[0])[0] + PROC_EXT
    if os.path.isfile(save_path) and not rebuild:
        return save_path
    with h5py.File(h5_path, "r") as hf:
        h, w = hf["height"].value, hf["width"].value
        assert h > 0 and w > 0
        person_skeletons = []
        for frm in range(len(hf) - 2):
            if "frame%d" % frm not in hf:
                warn("frame%d not exist in %s" % (frm, h5_path))
                continue
            joint_list = hf["frame%d" % frm]["joint_list"].value
            person_to_joint_assoc = hf["frame%d" % frm]["person_to_joint_assoc"].value
            canvas = decode_pose(joint_list, person_to_joint_assoc, (h, w), zoom_factor=zoom_factor)
            person_skeletons.append(canvas)
    person_skeletons = align_skeletons(person_skeletons)
    save_skeletons2video(person_skeletons, save_path, show_window=show_window)
    return save_path


def _walk_file(zoom_factor, show_window, rebuild, file_path):
    name, ext = os.path.splitext(file_path)
    if ext == ".h5":
        save_path = proc_h5(file_path, zoom_factor, show_window, rebuild)
        print("process {} into {}".format(file_path, save_path))
        return save_path
    else:
        return None


def prepare(root_dir, datalist_file, clslist_file, multiproc=True, zoom_factor=0.5, show_window=False, rebuild=False):
    sub_dirs = ["pos", "neg"]
    pool = Pool()
    os.makedirs(os.path.dirname(datalist_file), exist_ok=True)
    classes = []
    with open(datalist_file, 'w+', newline='') as f:
        writer = csv.writer(f)
        for sub_dir in sub_dirs:
            for root, dirs, files in os.walk(os.path.join(root_dir, sub_dir)):
                if len(files) == 0:
                    continue
                pa_func = partial(_walk_file,zoom_factor, show_window, rebuild)
                join_rets = pool.map(pa_func, [os.path.join(root, file) for file in files])
                for proc_path in join_rets:
                    if proc_path:
                        label = _get_label(sub_dir, os.path.basename(proc_path))
                        if label not in classes:
                            classes.append(label)
                        writer.writerow([label, proc_path])
    print("data list saved in %s" % datalist_file)
    pool.close()
    pool.join()
    classes = sorted(classes)
    os.makedirs(os.path.dirname(clslist_file), exist_ok=True)
    with open(clslist_file, "w+") as f:
        for cls in classes:
            f.write("%s\n" % cls)
    print("classes list saved in %s" % clslist_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="data root dir")
    parser.add_argument("data_list", type=str, help="csv file of dataset")
    parser.add_argument("cls_list", type=str, help="text file of classes list")
    args = parser.parse_args()
    prepare(args.root, args.data_list, args.cls_list)

