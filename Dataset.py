import copy
import csv
import functools
import glob
import os
import logging
import random
import math
from collections import namedtuple
import SimpleITK as sitk
import numpy as np
from dataclasses import dataclass, field

import torch
import torch.cuda
from torch.utils.data import Dataset
import torch.nn.functional as F


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
        # [1:]跳过标题列
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

        self.origin_xyz = xyzTuple(*ct_mhd.GetOrigin()) # 图像的原点坐标
        self.voxel_size = xyzTuple(*ct_mhd.GetSpacing())

        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)    # 获取方向向量并转化为（3， 3）的格式

    def get_candidate(self, center_xyz, width_irc):
        """从hu_a数组中获取一个以center_irc为中心的体素块，最后返回该体素块及中心坐标"""
        irc = xyz2irc(center_xyz, self.origin_xyz, self.voxel_size, self.direction_a)

        slice_list = []

        # 遍历irc的xyz三条轴
        for axis, center_val in enumerate(irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            # 检查中心是否在图像中
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.voxel_size, irc, axis])

            # 如果起始索引小于0，将起始索引设为0，结束索引为宽度
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            # 如果结束索引大于图像大小，将结束索引设为图像边界，起始索引设为边界-宽度
            if end_ndx > self.hu_a.shape[axis]:
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

def get_enhance_candidate(
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        use_cache=True):
    """
    对图像候选区域进行数据增强的函数
    :param augmentation_dict: 一个字典，指定增强类型和参数
    :param series_uid: ct序列的唯一标识符
    :param center_xyz: 结节的中心坐标
    :param width_irc: 结节的宽度
    :param use_cache: bool，决定是否使用缓存
    :return: 返回增强后的体素块（去除批次和通道维度，变为3D张量）和原始中心坐标center_irc。
    """
    if use_cache:
        ct_chunk, center_irc = \
            get_ct_candidate(series_uid, center_xyz, width_irc)
    else:
        ct = get_ct(series_uid)
        ct_chunk, center_irc = ct.get_candidate(center_xyz, width_irc)

    # 将ct_chunk从numpy数组转化为张量并增加两个维度
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    # 生成对角线为1的4x4张量，仿射变换的初始矩阵（核心）
    transform_t = torch.eye(4)

    # 遍历三个轴，随机进行翻转，偏移，缩放（前提是开展相应功能）
    # 注意是按轴随机
    for i in range(3):
        # 将仿射变换矩阵乘以-1
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        # 随机生成（-1，1）的随机数乘以偏移幅度，将其作为偏移量
        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float

        # 随机生成（-1，1）的随机数乘以缩放幅度最后加1，将其作为缩放大小
        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float

    # 进行随机角度的旋转
    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2

        # 计算sin和cos
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        # 创建旋转矩阵，使其图像围绕z轴旋转
        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    # 使用仿射变换矩阵生成仿射网格，表示每个输出体素对应的输入位置
    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32), # 获取仿射变换矩阵的前三行并添加批次维度
            ct_t.size(),    # 表示输出大小
            align_corners=False,    # 使用像素中心对齐
        )

    # 利用仿射网络对原始张量进行重采样
    augmented_chunk = F.grid_sample(
            ct_t,
            affine_t,
            padding_mode='border',  # 边界外使用边界值填充
            align_corners=False,
        ).to('cpu')

    # 添加高斯噪声
    if 'noise' in augmentation_dict:
        # 生成与augmented_chunk形状相同的标准正态分布随机张量并乘以我们设计的噪声强度
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        # 将噪声叠加到变换后的张量上
        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    """数据处理类"""
    def __init__(self,
                 val_stride=0,
                 is_val=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmentation_dict=None,
                 candidate_list=None,
            ):
        """

        :param val_stride: 用于指示分割数据集的步幅
        :param is_val: 指示是否为验证集
        :param series_uid: 表示特定的 CT 序列 ID。如果提供，则只处理该序列的数据，方便调试
        :param sortby_str:
        :param ratio_int:
        :param augmentation_dict:
        :param candidate_list:
        """
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidate_list:
            self.candidate_list = copy.copy(candidate_list)
            self.use_cache = False
        else:
            self.candidate_list = copy.copy(get_candidate_list())
            self.use_cache = True

        # 如果传入series_uid,那么这个类就是只针对该series_uid，方便进行调试
        if series_uid:
            self.candidate_list = [
                x for x in self.candidate_list if x.series_uid == series_uid
            ]

        if is_val:
            # 防止参数设置的有问题
            assert val_stride > 0, val_stride
            self.candidate_list = self.candidate_list[::val_stride]
            assert self.candidate_list
        elif val_stride > 0:
            del self.candidate_list[::val_stride]
            assert self.candidate_list

        if sortby_str == 'random':
            random.shuffle(self.candidate_list)
        elif sortby_str == 'series_uid':
            self.candidate_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [
            nt for nt in self.candidate_list if not nt.isNodule_bool
        ]
        self.pos_list = [
            nt for nt in self.candidate_list if nt.isNodule_bool
        ]

        logger.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidate_list),
            "validation" if is_val else "training",
            len(self.negative_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        """是Dataset类要求的，用于告诉DataLoader数据集包含多少个样本。"""
        if self.ratio_int:
            return 200000
        else:
            return len(self.candidate_list)

    def __getitem__(self, ndx):
        """根据索引ndx返回单个数据样本"""
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                candidate_tuple = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidate_tuple = self.pos_list[pos_ndx]
        else:
            candidate_tuple = self.candidate_list[ndx]

        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = get_enhance_candidate(
                self.augmentation_dict,
                candidate_tuple.series_uid,
                candidate_tuple.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = get_ct_candidate(
                candidate_tuple.series_uid,
                candidate_tuple.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = get_ct(candidate_tuple.series_uid)
            candidate_a, center_irc = ct.get_candidate(
                candidate_tuple.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
                not candidate_tuple.isNodule_bool,
                candidate_tuple.isNodule_bool
            ],
            dtype=torch.long,
        )

        return candidate_t, pos_t, candidate_tuple.series_uid, torch.tensor(center_irc)

