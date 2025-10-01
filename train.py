import argparse
import datetime
import sys
import logging
import warnings
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from Dataset import LunaDataset
from Model import LunaModel


# 设置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
    对输入进行迭代，并返回一个带索引的序列，同时提供进度跟踪的功能
    :param iter: 可迭代对象
    :param desc_str: 一句用于输出的描述语句
    :param start_ndx: 起始索引，表示从第几项开始迭代
    :param print_ndx: 下次输出日志的索引
    :param backoff: 控制日志输出的频率
    :param iter_len: 可迭代对象的长度
    """
    # 如果未传入可迭代对象则计算
    if iter_len is None:
        iter_len = len(iter)

    # 如果没有传入backoff则自行计算其值
    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    # 确保backoff大于2
    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    logger.warning("{} ----/{}, 迭代开始".format(
        desc_str,
        iter_len,
    ))

    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        # 使用yield关键字，将数据一步一步传出，避免一次性传出所有数据
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # 计算剩余还需时间
            duration_sec = ((time.time() - start_ts)    # 已经使用的时间
                            / (current_ndx - start_ndx + 1)    # 已经完成的数量
                            * (iter_len-start_ndx)    # 总数量
                            )

            # 将预计完成时间和剩余时间转化为可读的形式
            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            # -4表示左对齐占用4个字符
            logger.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0], # 转化为字符串，并去除微秒部分
                str(done_td).rsplit('.', 1)[0], # 同上
            ))

            # 更新下一次输出位置
            print_ndx *= backoff

    logger.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))


class LunaTrain:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.args = parser.parse_args()
        self.time_start = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.augmentation_dict = {}
        if self.args.augmented or self.args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.args.augmented or self.args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.args.augmented or self.args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.args.augmented or self.args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.args.augmented or self.args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        """初始化模型"""
        model = LunaModel()

        if torch.cuda.is_available():
            logger.info("Using CUDA; {} devices".format(torch.cuda.device_count()))

            # 如果设备有多个GPU，则使用nn.DataParallel类可以将工作分配给所有GPU
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            model = model.to(self.device)

        return model

    def init_optimizer(self):
        """初始化优化器"""
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def init_data(self, is_val):
        data_set = LunaDataset(
            val_stride=10,
            is_val=is_val,
            ratio_int=int(self.args.balanced),
            augmentation_dict=self.augmentation_dict,
        )

        batch_size = self.args.batch_size

        if torch.cuda.is_available():
            batch_size *= torch.cuda.device_count()

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=self.args.num_workers,
        )

        return data_loader

    def batch_loss(self, batch_ndx, batch_tuple, batch_size, metrics):
        """
        提取数据进行前向传播，并进行批次loss的运算
        :param batch_ndx:   当前批次的索引，用于跟踪当前处理的批次。
        :param batch_tuple:    包含输入数据、标签等信息的元组。
        :param batch_size:     每个批次的大小
        :param metrics:   存储指标
        :return: 返回批次loss的平均值
        """
        # input_t：数据；label_t：标签
        input_t, label_t, _series_list, _center_list = batch_tuple

        # 将数据转移到gpu上，non_blocking=True表示异步数据传输
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # 调用模型并自动进行前向传播，结果分别为未归一化的输出和归一化后的概率输出
        logits, probability = self.model(input_g)

        # reduction='none' 表示不对损失进行归约（即不求平均或求和），返回每个样本的损失值。
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(
            logits,
            label_g[:, 1],  # 提取第二列的标签
        )

        # 计算当前批次在metrics中的位置
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        # 从张量中提取真实标签，预测概率，loss并存取在metrics张量中。.detach()是创建一个新的副本并取消梯度，不影响原有张量
        metrics[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:, 1].detach()
        metrics[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability[:, 1].detach()
        metrics[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss.detach()

        return loss.mean()

    def log_metrics(self, epoch_nax, desc_str, metrics,):

        self.Threshold = 0.5

        label = metrics[METRICS_LABEL_NDX,:]
        prediction = metrics[METRICS_PRED_NDX,:]
        loss = metrics[METRICS_LOSS_NDX,:]

        pos_label = label >= self.Threshold
        neg_label = label < self.Threshold

        pos_pred = prediction >= self.Threshold
        neg_pred = prediction < self.Threshold

        neg_count = int(neg_label.sum())
        pos_count = int(pos_label.sum())

        true_neg_count = neg_correct = int((neg_label & neg_pred).sum())
        true_pos_count = pos_correct = int((pos_label & pos_pred).sum())

        false_pos_count = neg_count - neg_correct
        false_neg_count = pos_count - pos_correct


        accuracy = (true_neg_count + true_pos_count) / (neg_count + pos_count)
        precision = true_pos_count / (true_pos_count + false_pos_count)
        recall = true_pos_count / (true_pos_count + false_neg_count)

        logger.info("第{}个Epoch，{}的结果为，accuracy：{}；precision：{}；recall：{}".format(
            epoch_nax,
            desc_str,
            accuracy,
            precision,
            recall,
        ))

    def train(self, epoch_ndx, train_data):
        """
        基于训练集的模型前向传播和反向传播更新
        :param epoch_ndx: 当前训练循环数
        :param train_data:  训练集
        :return: 返回存储训练指标的张量并转移到cpu上
        """
        # 模型进入训练模式
        self.model.train()

        # 创建一个全零张量，存储训练阶段产生的指标
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,   # 表示要跟踪的指标数量
            len(train_data.dataset),    # 验证数据集的样本总数
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_data,
            "Epoch{} Training".format(epoch_ndx),
        )

        # # 将训练集转化为一个带索引的序列
        # batch_iter = enumerate(train_data)

        for batch_ndx, batch_tuple in batch_iter:
            # 清空优化器中的梯度，避免梯度累积
            self.optimizer.zero_grad()

            # 计算当前批次的loss
            loss = self.batch_loss(
                batch_ndx,
                batch_tuple,
                train_data.batch_size,
                trnMetrics_g    # 将张量传入，可以直接进行修改，并且不需要return出来
            )

            # 执行反向传播并更新模型参数
            loss.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.trn_writer.close()

        # 累加样本总量
        self.totalTrainingSamples_count += len(train_data.dataset)

        return trnMetrics_g.to('cpu')

    def test(self, epoch_ndx, val_data):
        """
        基于验证集进行模型评估，禁用模型更新
        :param epoch_ndx: 当前训练循环数
        :param val_data: 验证集
        :return: 返回存储模型指标的张量并转移到cpu上
        """
        # 禁用梯度计算，验证阶段不需要反向传播
        with torch.no_grad():
            # 模型进入评估模式，在当前模式会禁用如batchnorm和dropout等会影响模型稳定性的功能
            self.model.eval()

            # 创建一个全零张量，存储验证阶段产生的指标
            valMetrics_g = torch.zeros(
                METRICS_SIZE,   # 表示要跟踪的指标数量
                len(val_data.dataset),  # 验证数据集的样本总数
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_data,
                "Epoch{} Validation ".format(epoch_ndx),
            )

            # # 将验证集转化为一个带索引的序列
            # batch_iter = enumerate(val_data)

            for batch_ndx, batch_tuple in batch_iter:
                self.batch_loss(batch_ndx, batch_tuple, val_data.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def main(self):
        logger.info("开始{}，{}".format(type(self).__name__, self.args))

        train_data = self.init_data(is_val=False)
        val_data = self.init_data(is_val=True)

        for epoch_ndx in range(1, self.args.epochs + 1):

            logger.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.args.epochs,
                len(train_data),
                len(val_data),
                self.args.batch_size,
                (torch.cuda.device_count()),
            ))

            train_metrics_t = self.train(epoch_ndx, train_data)
            self.log_metrics(epoch_ndx, desc_str='训练集', metrics=train_metrics_t)

            val_metrics_t = self.test(epoch_ndx, val_data)
            self.log_metrics(epoch_ndx, desc_str='验证集', metrics=val_metrics_t)

        # if hasattr(self, 'trn_writer'):
        #     self.trn_writer.close()
        #     self.val_writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size',
                        help='Batch size to use for training',
                        default=32,
                        type=int,
                        )
    parser.add_argument('--num-workers',
                        help='Number of worker processes for background data loading',
                        default=0,
                        type=int,
                        )
    parser.add_argument('--epochs',
                        help='Number of epochs to train for',
                        default=10,
                        type=int,
                        )
    parser.add_argument('--balanced',
                        help="Balance the training data to half positive, half negative.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augmented',
                        help="Augment the training data.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-flip',
                        help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-offset',
                        help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-scale',
                        help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-rotate',
                        help="Augment the training data by randomly rotating the data around the head-foot axis.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-noise',
                        help="Augment the training data by randomly adding noise to the data.",
                        action='store_true',
                        default=False,
                        )


    LunaTrain().main()

