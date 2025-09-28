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

    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    logging.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            logging.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    logging.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))


class LunaTrain:

    def __init__(self, args=None):
        if sys.argv is None:
            sys.args = sys.argv[:1]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=0,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--tb-prefix',
                            default='p2ch11',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dwlpt',
                            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.args = parser.parse_args()
        self.time_start = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        """初始化模型"""
        model = LunaModel()

        if torch.cuda.is_available():
            logging.info("Using CUDA; {} devices".format(torch.cuda.device_count()))

            # 如果设备有多个GPU，则使用nn.DataParallel类可以将工作分配给所有GPU
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            model = model.to(self.device)

        return model

    def init_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def init_data(self, is_val):
        data_set = LunaDataset(
            val_stride=10,
            is_val=is_val,
        )

        batch_size = self.args.batch_size

        if torch.cuda.is_available():
            batch_size *= torch.cuda.device_count()

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        return data_loader

    def batch_loss(self, batch_ndx, batch_tuple, batch_size, train_measure):
        input_t, label_t, _series_list, _center_list = batch_tuple

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1],
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        train_measure[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:, 1].detach()
        train_measure[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:, 1].detach()
        train_measure[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean()

    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        self.initTensorboardWriters()
        logging.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict = {}
        metrics_dict['loss/all'] = \
            metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = \
            metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = \
            metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
            / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        logging.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        logging.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        logging.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        score = 1 \
            + metrics_dict['pr/f1_score'] \
            - metrics_dict['loss/mal'] * 0.01 \
            - metrics_dict['loss/all'] * 0.0001

        return score


    def train(self, epoch, train_data):

        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_data.dataset),
            device=self.device,
        )

        # batch_iter = enumerateWithEstimate(
        #     train_data,
        #     "E{} Training".format(epoch),
        #     start_ndx=train_data.num_workers,
        # )

        batch_iter = enumerate(train_data)

        print(len(train_data))

        a = 1

        for batch_ndx, batch_tup in batch_iter:

            print(f'第{a}次训练')

            a = a + 1

            self.optimizer.zero_grad()

            loss_var = self.batch_loss(
                batch_ndx,
                batch_tup,
                train_data.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.trn_writer.close()

        self.totalTrainingSamples_count += len(train_data.dataset)

        return trnMetrics_g.to('cpu')

    def test(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def main(self):
        logging.info("开始{}，{}".format(type(self).__name__, self.args))

        train_data = self.init_data(is_val=False)
        val_data = self.init_data(is_val=True)

        for epoch_ndx in range(1, self.args.epochs + 1):

            logging.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.args.epochs,
                len(train_data),
                len(val_data),
                self.args.batch_size,
                (torch.cuda.device_count()),
            ))

            trnMetrics_t = self.train(epoch_ndx, train_data)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.test(epoch_ndx, val_data)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()


if __name__ == '__main__':
    LunaTrain().main()










