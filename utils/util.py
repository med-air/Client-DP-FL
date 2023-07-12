from scipy import ndimage
import numpy as np
from medpy import metric
import logging
import os
import time
import torch


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}.log".format(logger_name))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def _connectivity_region_analysis(mask):
    s = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    label_im, nb_labels = ndimage.label(mask)  # , structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    # print(sizes)
    # print(np.argmax(sizes))

    # plt.imshow(label_im)
    # plt.show()
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    # plt.imshow(label_im)
    # plt.show()

    return label_im


def _eval_haus(pred, gt):
    """
    :param pred: whole brain prediction
    :param gt: whole
    :param detail:
    :return: a list, indicating Dice of each class for one case
    """
    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().numpy()
    pred[pred >= 0.5] = 1.0
    pred[pred < 0.5] = 0.0
    gt = gt.detach().cpu().numpy()
    # haus = []

    """ During the first several epochs, prediction may be all zero, which will throw an error. """
    if pred.sum() == 0:
        hd = torch.tensor(1000.0)
    else:
        hd = metric.binary.hd95(pred, gt)
    # hd = metric.binary.hd(gt, pred)

    return hd


def _eval_iou(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5, smooth=1e-5):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = (outputs > threshold).long()
    labels = labels.long()
    intersection = (
        (outputs & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0

    thresholded = (
        torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds

    return iou.mean()


def cos_sim(a, b):
    from numpy import dot
    from numpy.linalg import norm

    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

