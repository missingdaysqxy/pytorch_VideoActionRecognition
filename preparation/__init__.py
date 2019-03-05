# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 16:29
# @Author  : liuqixuan_i
# @Email   : liuqixuan_i@didiglobal.com
# @File    : __init__.py
# @Software: PyCharm

from preparation.pose import decode_pose,align_skeletons
from preparation.prepare_data import prepare
try:
    import ipdb
except:
    import pdb as ipdb
