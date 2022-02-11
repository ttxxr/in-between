import torch
from torch import nn
import numpy as np
from .quanternion import *
import matplotlib.pyplot as plt


def PLU(x, alpha=0.1, c=1.0):
    relu = nn.ReLU()
    o1 = alpha * (x + c) - c
    o2 = alpha * (x - c) + c
    o3 = x - relu(x - o2)
    o4 = relu(o1 - o3) + o3
    return o4


def cal_ADE_metric(pred, gt):
    """
    Average displacement error
    :param pred: pred global position, TxJx3
    :param gt: gt global position
    :return: ADE
    """
    diff = pred - gt
    ade = np.linalg.norm(diff, ord=2, axis=2).mean()
    return ade


def cal_FDE_metric(pred, gt):
    """
    Final displacement error
    :param pred: pred global position, TxJx3
    :param gt: gt global position
    :return:
    """
    last_diff = pred[-1] - gt[-1]
    fde = np.linalg.norm(last_diff, ord=2, axis=1).mean()
    return fde


def gen_ztta(distance=50, dim=256, basis=10000):
    ztta = np.zeros((1, distance, dim))
    for i in range(distance):
        time_to_arrive = distance - i
        for d in range(dim):
            if dim % 2 == 0:
                ztta[:, i, d] = np.sin(time_to_arrive / np.power(basis, d / dim))
            else:
                ztta[:, i, d] = np.cos(time_to_arrive / np.power(basis, (d - 1) / dim))

    return torch.from_numpy(ztta.astype(np.float_))  # 1x50x256


def gen_batch_ztta(len_list, distance=50, dim=256, basis=10000):
    # ztta = torch.zeros((len_list.shape[0], distance, dim))
    ztta = np.zeros((len_list.shape[0], distance, dim))
    for i in range(distance):
        time_to_arrive = len_list - i
        for d in range(dim):
            if dim % 2 == 0:
                # ztta[:, i, d] = torch.sin(time_to_arrive / torch.pow(basis, time_to_arrive / dim))
                ztta[:, i, d] = np.sin(time_to_arrive / np.power(basis, time_to_arrive / dim))
            else:
                # ztta[:, i, d] = torch.cos(time_to_arrive / torch.pow(basis, (time_to_arrive - 1) / dim))
                ztta[:, i, d] = np.cos(time_to_arrive / np.power(basis, (time_to_arrive - 1) / dim))

    # return ztta
    return torch.from_numpy(ztta.astype(np.float_))  # Bx50x256


def gen_ztarget(steps, distance=50):
    time_to_arrive = distance - steps - 1
    if time_to_arrive < 5:
        lambda_target = 0
    elif time_to_arrive >= 30:
        lambda_target = 1
    else:
        lambda_target = (time_to_arrive - 5) / 25

    return lambda_target


def gen_ztarget_by_tta(tta):
    time_to_arrive = tta
    if time_to_arrive < 5:
        lambda_target = 0
    elif time_to_arrive >= 30:
        lambda_target = 1
    else:
        lambda_target = (time_to_arrive - 5) / 25

    return lambda_target


def gen_ratio(steps, distance=50):
    time_to_arrive = distance - steps - 1
    if time_to_arrive < 5:
        lambda_target = 0.7
    elif time_to_arrive >= 45:
        lambda_target = 0
    else:
        lambda_target = (time_to_arrive - 5) / 40

    return lambda_target


def gen_target_loss_weight(steps, distance=50):
    time_to_arrive = distance - steps - 1
    if time_to_arrive < 5:
        lambda_target = 1 - time_to_arrive / 5
    else:
        lambda_target = 0

    return lambda_target


def write_to_bvhfile(data, filename, joints_to_remove):
    fout = open(filename, 'w')
    line_cnt = 0
    for line in open('./example.bvh', 'r'):
        fout.write(line)
        line_cnt += 1
        if line_cnt >= 132:
            break
    fout.write(('Frames: %d\n' % data.shape[0]))
    fout.write('Frame Time: 0.033333\n')
    pose_data = qeuler_np(data[:, 3:].reshape(data.shape[0], -1, 4), order='zyx', use_gpu=False)
    pose_data = pose_data / np.pi * 180.0
    for t in range(data.shape[0]):
        line = '%f %f %f ' % (data[t, 0], data[t, 1], data[t, 2])
        for d in range(pose_data.shape[1] - 1):
            line += '%f %f %f ' % (pose_data[t, d, 2], pose_data[t, d, 1], pose_data[t, d, 0])
        line += '%f %f %f\n' % (pose_data[t, -1, 2], pose_data[t, -1, 1], pose_data[t, -1, 0])
        fout.write(line)
    fout.close()


def plot_pose(pose, cur_frame, prefix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    ax.cla()
    num_joint = pose.shape[0] // 3
    for i, p in enumerate(parents):
        if i > 0:
            ax.plot([pose[i, 0], pose[p, 0]],
                    [pose[i, 2], pose[p, 2]],
                    [pose[i, 1], pose[p, 1]], c='r')

            # if i == 3:
            #     ax.plot([pose[i + num_joint, 0], pose[p + num_joint, 0]],
            #             [pose[i + num_joint, 2], pose[p + num_joint, 2]],
            #             [pose[i + num_joint, 1], pose[p + num_joint, 1]], c='r')
            # else:
            #     ax.plot([pose[i + num_joint, 0], pose[p + num_joint, 0]],
            #             [pose[i + num_joint, 2], pose[p + num_joint, 2]],
            #             [pose[i + num_joint, 1], pose[p + num_joint, 1]], c='b')

            ax.plot([pose[i + num_joint, 0], pose[p + num_joint, 0]],
                    [pose[i + num_joint, 2], pose[p + num_joint, 2]],
                    [pose[i + num_joint, 1], pose[p + num_joint, 1]], c='b')
            ax.plot([pose[i + num_joint * 2, 0], pose[p + num_joint * 2, 0]],
                    [pose[i + num_joint * 2, 2], pose[p + num_joint * 2, 2]],
                    [pose[i + num_joint * 2, 1], pose[p + num_joint * 2, 1]], c='g')
    xmin = np.min(pose[:, 0])
    ymin = np.min(pose[:, 2])
    zmin = np.min(pose[:, 1])
    xmax = np.max(pose[:, 0])
    ymax = np.max(pose[:, 2])
    zmax = np.max(pose[:, 1])
    scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
    xmid = (xmax + xmin) // 2
    ymid = (ymax + ymin) // 2
    zmid = (zmax + zmin) // 2
    ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
    ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
    ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

    plt.draw()
    plt.savefig(prefix + '_' + str(cur_frame) + '.png', dpi=200, bbox_inches='tight')
    plt.close('all')


def plot_pose_mtarget(pose, target_num, cur_frame, prefix):
    """
    可打印多个target帧，所有target位于pose的末尾
    :param pose:
    :param target_num:
    :param cur_frame:
    :param prefix:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    ax.cla()
    num_joint = pose.shape[0] // (1 + 1 + target_num)
    for i, p in enumerate(parents):
        if i > 0:
            ax.plot([pose[i, 0], pose[p, 0]],
                    [pose[i, 2], pose[p, 2]],
                    [pose[i, 1], pose[p, 1]], c='r')
            ax.plot([pose[i + num_joint, 0], pose[p + num_joint, 0]],
                    [pose[i + num_joint, 2], pose[p + num_joint, 2]],
                    [pose[i + num_joint, 1], pose[p + num_joint, 1]], c='b')

            for t in range(target_num):
                ax.plot([pose[i + num_joint * (2 + t), 0], pose[p + num_joint * (2 + t), 0]],
                        [pose[i + num_joint * (2 + t), 2], pose[p + num_joint * (2 + t), 2]],
                        [pose[i + num_joint * (2 + t), 1], pose[p + num_joint * (2 + t), 1]], c='g')

    xmin = np.min(pose[:, 0])
    ymin = np.min(pose[:, 2])
    zmin = np.min(pose[:, 1])
    xmax = np.max(pose[:, 0])
    ymax = np.max(pose[:, 2])
    zmax = np.max(pose[:, 1])
    scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
    xmid = (xmax + xmin) // 2
    ymid = (ymax + ymin) // 2
    zmid = (zmax + zmin) // 2
    ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
    ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
    ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

    plt.draw()
    plt.savefig(prefix + '_' + str(cur_frame) + '.png', dpi=200, bbox_inches='tight')
    plt.close('all')
