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
from multiprocessing import Pool
from functools import partial
from pose import decode_pose, align_skeletons

_PROC_EXT_ = ".proch5.mp4"
_REGEX_ = re.compile("^(?P<seq>\d+)_(?P<label>\w+)_(?P<num>\d+){}$".format(_PROC_EXT_.replace(".", r"\.")))
assert os.path.splitext(_PROC_EXT_)[1] in [".mp4"], "must save as .mp4 video file!"


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
                cv2.destroyAllWindows()
                break
    cv2.destroyAllWindows()
    out.release()


def proc_h5(h5_path, zoom_factor=1.0, show_window=True, rebuild=False):
    save_path = os.path.splitext(os.path.splitext(h5_path)[0])[0] + _PROC_EXT_
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
    if multiproc:
        pool = Pool()
    os.makedirs(os.path.dirname(datalist_file), exist_ok=True)
    classes = []
    count = 0
    with open(datalist_file, 'w+', newline='') as f:
        writer = csv.writer(f)
        for sub_dir in sub_dirs:
            for root, dirs, files in os.walk(os.path.join(root_dir, sub_dir)):
                if len(files) == 0:
                    continue
                if multiproc:
                    parad_func = partial(_walk_file, zoom_factor, show_window, rebuild)
                    join_rets = pool.map(parad_func, [os.path.join(root, file) for file in files])
                else:
                    join_rets = []
                    for path in [os.path.join(root, file) for file in files]:
                        join_rets.append(_walk_file(zoom_factor, show_window, rebuild, path))
                for proc_path in join_rets:
                    if proc_path:
                        label = _get_label(sub_dir, os.path.basename(proc_path))
                        if label not in classes:
                            classes.append(label)
                        writer.writerow([label, proc_path])
                        count += 1
    print("data list saved in %s" % datalist_file)
    if multiproc:
        pool.close()
        pool.join()
    classes = sorted(classes)
    os.makedirs(os.path.dirname(clslist_file), exist_ok=True)
    with open(clslist_file, "w+") as f:
        for cls in classes:
            f.write("%s\n" % cls)
    print("classes list saved in %s" % clslist_file)
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="data root dir")
    parser.add_argument("data_list", type=str, help="csv file save path of dataset")
    parser.add_argument("cls_list", type=str, help="text file save path of classes list")
    parser.add_argument("-z", "--zoom", type=float, default=0.5, help="zoom factor for .proch5.mp4 skeleton videos")
    parser.add_argument("-v", "--verb", action="store_true", help="show visualized skeletons video window")
    parser.add_argument("-s", "--singleproc", action="store_true", help="use single-process, do not use multi-process")
    parser.add_argument("-b", "--rebuild", action="store_true", help="rebuild existed proch5.mp4 skeleton videos")
    args = parser.parse_args()
    prepare(args.root, args.data_list, args.cls_list, multiproc=not args.singleproc, zoom_factor=args.zoom,
            show_window=args.verb, rebuild=args.rebuild)
