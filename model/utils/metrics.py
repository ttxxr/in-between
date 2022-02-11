import torch
from torch import nn
import numpy as np
from .quanternion import *


def cal_ADE_FDE_metric(pred, gt):
    """
    Average displacement error & Final displacement error
    :param pred: pred global position, BxTxJx3
    :param gt: gt global position
    :return: ADE, FDE
    """
    diff = pred - gt
    norm = torch.linalg.norm(diff, ord=2, axis=3).mean(axis=2)
    ADE = norm.mean(axis=1).mean()/100
    FDE = norm[:, -1].mean()/100

    return np.asarray(ADE.cpu()), np.asarray(FDE.cpu())


def cal_F5DE_metric(pred, gt):
    diff = pred - gt
    norm = torch.linalg.norm(diff, ord=2, axis=3).mean(axis=2)
    F5DE = norm[:, -5:].mean()/100

    return np.asarray(F5DE.cpu())
