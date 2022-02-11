import numpy as np
import imageio
import torch
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image


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


def blend(normal_gp, inverse_gp):
    weight_pred_list = []
    blend_left_border = 5
    blend_right_border = 45
    for i in range(50):
        if i < blend_left_border:
            weight = 1
        elif i >= blend_right_border:
            weight = 0
        else:
            weight = 1 - (i - blend_left_border) / blend_right_border
            # weight = 0.7
        weight_pred_list.append(weight)
        normal_gp[i] = normal_gp[i] * weight + inverse_gp[i] * (1 - weight)
    return normal_gp


batch_idx = "102"
inverse_gt = torch.tensor(np.load("./npz/" + batch_idx + "inverse_gt.npz")["arr_0"])
inverse_pred = np.load("./npz/" + batch_idx + "inverse_pred.npz")["arr_0"][::-1, :, :, :]
inverse_pred = torch.tensor(inverse_pred.copy())

gt = torch.tensor(np.load("./npz/" + batch_idx + "gt.npz")["arr_0"])
pred = torch.tensor(np.load("./npz/" + batch_idx + "pred.npz")["arr_0"])

global_position = gt
bs = 6
img_list = []

# blend
pred = blend(pred, inverse_pred)

for t in range(50):
    pred_global_position = pred[t]
    plot_pose(np.concatenate([global_position[bs, 0].view(22, 3).detach().cpu().numpy(),
                              pred_global_position[bs].view(22, 3).detach().cpu().numpy(),
                              global_position[bs, -1].view(22, 3).detach().cpu().numpy()], 0),
              t, './results/img/pred')
    k = t if t < 50 else 49
    plot_pose(np.concatenate([global_position[bs, 0].view(22, 3).detach().cpu().numpy(),
                              # global_position[bs, t + 1].view(22, 3).detach().cpu().numpy(),
                              global_position[bs, k].view(22, 3).detach().cpu().numpy(),
                              global_position[bs, -1].view(22, 3).detach().cpu().numpy()], 0),
              t, './results/img/gt')
    pred_img = Image.open('./results/img/pred_' + str(t) + '.png', 'r')
    gt_img = Image.open('./results/img/gt_' + str(t) + '.png', 'r')
    img_list.append(np.concatenate([pred_img, gt_img.resize(pred_img.size)], 1))

# save gif
imageio.mimsave("./npz/merged/" + batch_idx + "img.gif", img_list, duration=0.1)
