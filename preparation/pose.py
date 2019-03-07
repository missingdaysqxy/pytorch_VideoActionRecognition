# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 12:44
# @Author  : liuqixuan_i
# @Email   : liuqixuan_i@didiglobal.com
# @File    : pose.py
# @Software: PyCharm

import cv2
import math
import numpy as np

# Heatmap indices to find each limb (joint connection). Eg: limb_type=1 is
# Neck->LShoulder, so joint_to_limb_heatmap_relationship[1] represents the
# indices of heatmaps to look for joints: neck=1, LShoulder=5
joint_to_limb_heatmap_relationship = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]

NUM_LIMBS = len(joint_to_limb_heatmap_relationship)


def decode_pose(joint_list, person_to_joint_assoc, img_shape, zoom_factor):
    # type:(tuple,float,np.ndarray,np.ndarray)->np.ndarray
    limb_thickness = max(int(2e-3 * zoom_factor * sum(img_shape[0:2])), 1)
    point_radius = limb_thickness * 2
    canvas = np.zeros((int(zoom_factor * img_shape[0]), int(zoom_factor * img_shape[1]), 3), dtype=np.float)
    for person_id, person_joint_info in enumerate(person_to_joint_assoc):
        sub_canvas = np.zeros_like(canvas)
        color_p = person_id + 1
        for limb_idx in range(NUM_LIMBS):
            color_j = limb_idx + 1
            joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_idx]].astype(int)
            if -1 in joint_indices:
                # Only draw actual limbs (connected joints), skip if not connected
                continue
            # joint_coords[:,0] represents Y coords of both joints; joint_coords[:,1], X coords
            joint_coords = zoom_factor * joint_list[joint_indices, 0:2]

            for joint in joint_coords:  # Draw circles at every joint
                cv2.circle(sub_canvas, tuple(joint[:2].astype(int)), point_radius, [color_p, color_j, 0],
                           thickness=-1)
            # mean along the axis=0 computes meanYcoord and meanXcoord -> Round and make int to avoid errors
            coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
            # joint_coords[0,:] is the coords of joint_src; joint_coords[1,:] is the coords of joint_dst
            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            # Get the angle of limb_dir in degrees using atan2(limb_dir_x, limb_dir_y)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))

            polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(sub_canvas, polygon, [color_p, 0, color_j])
        canvas += sub_canvas
    return canvas


def align_skeletons(skeletons_list):
    # ToDo: make this work..
    return skeletons_list

