import copy
import csv
import functools
import glob
import os
import logging
from collections import namedtuple
import SimpleITK as sitk
import numpy as np
from dataclasses import dataclass, field

import torch
import torch.cuda
from torch.utils.data import Dataset


# 设置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# 创建轻量级的、不可变的类，类似于元组，但每个元素可以通过名称访问，适合存储结构化的数据。
CandidateTuple = namedtuple(
    'CandidateTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)
IrcTuple = namedtuple('IrcTuple', ['index', 'row', 'col'])
xyzTuple = namedtuple('xyzTuple', ['x', 'y', 'z'])


def irc2xyz(coord_irc, origin_xyz, voxel_size, direction_a):
    """
    将irc坐标转化为xyz坐标
    coord_irc : 输入的irc坐标
    origin_xyz : xyz坐标原点
    voxel_size : 表示体素的大小，一个三元组，表示其在xyz方向上的大小
    direction_a : 表示坐标变化时的方向旋转
    """
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    voxel_size_a = np.array(voxel_size)

    # 先得到相对原点的偏移，然后旋转，最后加上原点坐标
    coords_xyz = (direction_a @ (cri_a * voxel_size_a)) + origin_a

    return xyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, voxel_size, direction_a):
    """ 将xyz坐标转化为irc坐标 """
    origin_a = np.array(origin_xyz)
    voxel_size_a = np.array(voxel_size)
    coord_a = np.array(coord_xyz)

    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / voxel_size_a
    cri_a = np.round(cri_a)    # 对结果进行四舍五入

    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


@functools.lru_cache(1)     # 最近最少使用缓存，即保存最近一次该函数运行的结果（区分参数的）
def get_candidate_list(only_download = True):
    """获取所有节点位置，并且添加数据部分下载的处理功能"""
    # glob.glob()的作用是返回所有符合要求的文件路径列表
    mhd_list = glob.glob('data/luna/subset*/*.mhd')
    downloaded_data = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # 处理annotations.csv文件
    diameter_dict = {}
    with open('data/annotations.csv', "r") as f:
        # [1:]跳过标题行
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # row前开后闭，tuple生成元组，即转化为坐标
            annotation_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter = float(row[4])

            # setdefault(key, default)是字典的方法，如果key不在字典中，则为该键设置一个默认值，返回该值
            # 将检索出的数据整理到字典中，多个值以列表表示
            diameter_dict.setdefault(series_uid, []).append((annotation_xyz, annotation_diameter))

    # 处理candidates.csv文件
    candidate_list = []
    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # 如果一个样本的series_uid不在数据中，说明该样本没有被收录，需要跳过
            if series_uid not in downloaded_data and only_download:
                continue

            # 获取第5列class的数据，表示该位置是否是结节
            is_nodule = bool(int(row[4]))
            candidate_xyz = tuple([float(x) for x in row[1:4]])

            #表示尚未找到结节的直径
            candidate_diameter = 0.0
            # 获取键为series_uid时的值，如果没有该键则返回空列表
            for annotation_tup in diameter_dict.get(series_uid, []):
                # 解包 annotation_tup 元组
                annotation_xyz, annotation_diameter = annotation_tup
                for i in range(3):
                    delta_mm = abs (candidate_xyz[i] - annotation_xyz[i])
                    # 4分之一的直径是主观设计的，如果差距较大则表示不一致
                    if delta_mm > annotation_diameter / 4:
                        continue
                    else:
                        candidate_diameter = annotation_diameter
                        break

            candidate_list.append(CandidateTuple(
                is_nodule,  #表示该位置是否是结节
                candidate_diameter,  # 表示直径
                series_uid,  # 表示结节编码
                candidate_xyz,  #以元组表示结节位置
            ))

    # 进行降序排序
    candidate_list.sort(reverse=True)

    return candidate_list


class CT:
    """一个专门为CT数据设计的类"""
    def __init__(self, series_uid):
        mhb_path = glob.glob('data/luna/subset*/{}.mhd'.format(series_uid))[0]

        # 读取数据为simpleITK对象，并转化为numpy数组
        ct_mhd = sitk.ReadImage(mhb_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), np.float32)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = xyzTuple(*ct_mhd.GetOrigin())
        self.voxel_size = xyzTuple(*ct_mhd.GetSpacing())

        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)    # 获取方向向量并转化为（3， 3）的格式

    def get_candidate(self, center_xyz, width_irc):
        """从hu_a数组中获取一个以center_irc为中心的体素块，最后返回该体素块及中心坐标"""
        irc = xyz2irc(center_xyz, self.origin_xyz, self.voxel_size, self.direction_a)

        slice_list = []

        for axis, center_val in enumerate(irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.voxel_size, irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, irc


@ functools.lru_cache(1, typed=True)    # 最近最少使用缓存，即保存最近一次该函数运行的结果（区分参数的）
def get_ct(series_uid):
    """利用缓存，避免重复加载数据"""
    return CT(series_uid)

# @ raw_cache.memoize(typed=True)
def get_ct_candidate(series_uid, center_xyz, width_irc):
    """主要作用同上，但是用于提取候选区域"""
    ct = get_ct(series_uid)
    ct_chunk, center_irc = ct.get_candidate(center_xyz, width_irc)

    return ct_chunk, center_irc


class LunaDataset(Dataset):
    """数据处理类"""
    def __init__(self, val_stride=0, is_val=None, series_uid=None):
        """
        val_stride: 用于指示分割数据集的步幅
        is_val: 指示是否为验证集
        series_uid: 表示特定的 CT 序列 ID。如果提供，则只处理该序列的数据，方便调试
        """
        # 复制结果，防止修改对缓存副本造成影响
        self.candidate_list = copy.copy(get_candidate_list())

        # 如果传入series_uid,那么这个类就是只针对该series_uid，方便进行调试
        if series_uid:
            self.candidate_list = {
                x for x in self.candidate_list if x.series_uid == series_uid
            }

        if is_val:
            # 防止参数设置的有问题
            assert val_stride > 0, val_stride
            self.candidate_list = self.candidate_list[::val_stride]
            assert self.candidate_list
        elif val_stride > 0:
            del self.candidate_list[::val_stride]
            assert self.candidate_list

    def __len__(self):
        """是Dataset类要求的，用于告诉DataLoader数据集包含多少个样本。"""
        return len(self.candidate_list)

    def __getitem__(self, ndx):
        """根据索引ndx返回单个数据样本"""
        candidate_tuple = self.candidate_list[ndx]
        width_irc = (32, 48, 48)

        # 加载CT候选数据块
        candidate_a, center_irc = get_ct_candidate(
            candidate_tuple.series_uid,
            candidate_tuple.center_xyz,
            width_irc,
        )

        # 将数据从numpy数组转化为tensor
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)  # 增加通道维度

        # pos_t 是一个二分类标签，表示候选区域是否为结节
        pos_t = torch.tensor([
            not candidate_tuple.isNodule_bool,  # 非结节的概率
            candidate_tuple.isNodule_bool,  # 结节的概率
        ],
        dtype = torch.long,
        )

        return (
            candidate_t,    # 处理后的CT数据张量
            pos_t,  #标签
            candidate_tuple.series_uid,     # 序列 ID
            torch.tensor(center_irc),   # 中心坐标
        )


