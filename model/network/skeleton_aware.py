import torch.nn as nn
import torch
import numpy as np

"""
Since I didn't find an elegant way to find the topology every time after Skeleton pooling
I use a dirty trick to hard-encode the topologies after pooling
So, this means we could only apply our model on Aj and Bigvegas's skeletons
"""
topology_after_1_pool = np.array([0, 0, 1, 0, 3, 0, 5, 6, 6, 8, 6, 10])
ee_id_after_1_pool = [2, 4, 7, 9, 11]

topology_after_2_pool = np.array([0, 0, 0, 0, 3, 3, 3])
ee_id_after_2_pool = [1, 2, 4, 5, 6]


class SkeletonConvolution(nn.Module):
    """
    The skeleton convolution based on the paper
    Use a more intuitive 2D convolution than the 1D convolution in the original source code
    """

    def __init__(self, in_channels, out_channels, k_size: tuple, stride, pad_size: tuple,
                 topology: np.ndarray, neighbor_dist: int, ee_id: list):
        """
        :param k_size: should be (simple_joint_num, a_short_time_length) !!!
        :param topology: A numpy array, the value of the array is parent idx, the idx of the array is child node idx
        :param ee_id: end effectors index
        it could tell us the topology of the initial simplified skeleton or the pooled skeleton
        """
        super(SkeletonConvolution, self).__init__()
        # find neighbors in neighbor_distance
        self.neighbor_list = find_neighbor(topology, neighbor_dist, ee_id)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad_size)

    def forward(self, x):
        """
        :param x: The input tensor should have size [B , IN_C , simple_joint_num , total_time_length]
        :return:
        """
        result_list = []
        # TODO：每一帧、每一个节点分别做conv，开销很大
        for neighbors in self.neighbor_list:
            binary_mask = torch.zeros_like(x, dtype=torch.float)
            # only neighboring joint can has mask value 1
            binary_mask[:, :, neighbors, :] = 1
            tmp_x = x * binary_mask
            # tmp_result should have size [B * OUT_C * 1 * total_time_length]
            tmp_result = self.conv(tmp_x)
            result_list.append(tmp_result)
        return torch.cat(result_list, dim=2)


class SkeletonPool(nn.Module):
    """
    Apply average skeleton pooling on the output of the skeleton convolution layers
    The pooling layer should tell that: what's the topology of the next convolution layer!.
    """

    def __init__(self, topology, ee_id, layer_idx):
        """
        :param topology: 1D numpy array
        :param ee_id: A list
        """
        super(SkeletonPool, self).__init__()
        self.old_topology = topology
        self.old_ee_id = ee_id
        # store the topology after pooling and merge joints
        self.new_topology = topology_after_1_pool if layer_idx == 1 else topology_after_2_pool
        # store the ee_ids after pooling and merge joints
        self.new_ee_id = ee_id_after_1_pool if layer_idx == 1 else ee_id_after_2_pool
        self.seq_list = []
        self.pooling_list = []
        self.old_joint_num = len(self.old_topology.tolist())
        self.new_joint_num = len(self.new_topology.tolist())
        # calculate the degree of each joint in the skeleton graph
        # 经过下面的操作，计算各个joint的degree
        self.degree = calculate_degree(topology)
        # separate the skeleton into multiple sequence
        self.pooling_seq = find_pool_seq(self.degree)  # a list
        self.merge_pairs = self._get_merge_pairs()  # a list
        self.merge_nums = [len(each) for each in self.merge_pairs]

    def _get_merge_pairs(self):
        merge_pair_list = []
        for seq in self.pooling_seq:
            if len(seq) == 1:
                single_joint = [seq.pop(0)]
                merge_pair_list.append(single_joint)
                continue
            elif len(seq) % 2 != 0:
                single_joint = [seq.pop(0)]
                merge_pair_list.append(single_joint)
            for i in range(0, len(seq), 2):
                tmp_pair = [seq[i], seq[i + 1]]
                merge_pair_list.append(tmp_pair)
        return merge_pair_list

    def forward(self, x):
        result_list = []
        result_list.append(x[:, :, 0:1, :])  # add the root joint's data into result
        for merge_pair in self.merge_pairs:
            tmp_result = torch.zeros_like(x[:, :, 0:1, :])
            for merge_idx in merge_pair:
                tmp_result += x[:, :, merge_idx: merge_idx + 1, :]
            tmp_result /= len(merge_pair)
            result_list.append(tmp_result)
        result = torch.cat(result_list, dim=2)
        if result.shape[2] != self.new_joint_num:
            raise Exception('Joint num does not match after pooling')
        return result


class SkeletonUnPool(nn.Module):
    def __init__(self, un_pool_expand_nums: list):
        """
        :param un_pool_expand_nums: 一个列表，记录着对应的pooling层，每个merge后的joint是由几个关节合并得到的。
        由几个关节merge得到，在UnPool的时候，就把该关节的tensor复制几次。
        需要注意的是，root joint是从未被merge过的，所以也不duplicate它。
        """
        super(SkeletonUnPool, self).__init__()
        self.un_pool_expand_nums = un_pool_expand_nums

    def forward(self, x):
        result_list = []
        result_list.append(x[:, :, 0:1, :])  # add root joint's feature tensor first
        for idx, expand_num in enumerate(self.un_pool_expand_nums):
            tmp_idx = idx + 1
            tmp_x = x[:, :, tmp_idx: tmp_idx + 1, :]
            tmp_x = tmp_x.repeat(1, 1, expand_num, 1)
            result_list.append(tmp_x)
        out = torch.cat(result_list, dim=2)
        return out


def build_bone_topology(topology):
    # The topology is simplified already!
    # get all edges (parents_bone_idx, current_bone_idx)
    # edges 要比topology的个数少1
    edges = []
    joint_num = len(topology)
    # 舍去了root joint
    for i in range(1, joint_num):
        # i 指的是简化后骨骼的index, topology[i] is i's parents bone
        edges.append((topology[i], i))
    return edges


def calculate_neighbor_matrix(topology):
    # topology = topology.tolist()
    joint_num = len(topology)
    mat = [[100000] * joint_num for _ in range(joint_num)]
    for i, j in enumerate(topology):
        mat[i][j] = 1
        mat[j][i] = 1
    for i in range(joint_num):
        mat[i][i] = 0
    # Floyd algorithm to calculate distance between nodes of the skeleton graph
    for k in range(joint_num):
        for i in range(joint_num):
            for j in range(joint_num):
                mat[i][j] = min(mat[i][j], mat[i][k] + mat[k][j])
    return mat


def calculate_degree(topology):
    # topology = topology.tolist()
    joint_num = len(topology)
    mat = [[0] * joint_num for _ in range(joint_num)]
    for i, j in enumerate(topology):
        mat[i][j] = 1
        mat[j][i] = 1
    for i in range(joint_num):
        mat[i][i] = 0
    degree_list = [sum(each) for each in mat]
    return degree_list


def find_neighbor(topology, dist, ee_id):
    distance_mat = calculate_neighbor_matrix(topology)
    neighbor_list = []
    joint_num = len(distance_mat)
    for i in range(joint_num):
        neighbor = []
        for j in range(joint_num):
            # 距离小于d的，就加入到该骨骼的领接列表中来
            if distance_mat[i][j] <= dist:
                neighbor.append(j)
        # 将每根骨骼的邻接列表插入到一个总的列表中
        neighbor_list.append(neighbor)

    # add neighbor for global part(the root joint's neighbors' index)
    global_part_neighbor = neighbor_list[0].copy()
    # based on the paper, the end_effector should also be regarded as the neighbor of the root joint
    # so we need to add end_effector's index to root joint's neighbor list
    global_part_neighbor = list(set(global_part_neighbor).union(set(ee_id)))
    # 互相添加为邻居节点
    for i in global_part_neighbor:
        # the index of root joint is 0!
        if 0 not in neighbor_list[i]:
            neighbor_list[i].append(0)
    neighbor_list[0] = global_part_neighbor
    return neighbor_list


def find_pool_seq(degree):
    num_joint = len(degree)
    seq_list = [[]]
    for joint_idx in range(1, num_joint):
        if degree[joint_idx] == 2:
            seq_list[-1].append(joint_idx)
        else:
            seq_list[-1].append(joint_idx)
            seq_list.append([])
            continue
    seq_list = [each for each in seq_list if len(each) != 0]
    return seq_list


class EncBasicBlock(nn.Module):
    """
    The Convolution + ReLU + Pooling block for building Encoder(both dynamic or static)
    """

    def __init__(self, args, in_channel, out_channel, topology, ee_id, layer_idx, dynamic=True):
        super(EncBasicBlock, self).__init__()
        joint_num = len(topology)

        kernel_size = (joint_num, args["dynamic_kernel_size"]) if dynamic else (joint_num, args["static_kernel_size"])
        hidden_channel = out_channel // 2
        self.conv1by1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)

        self.conv1 = SkeletonConvolution(in_channels=in_channel,
                                         out_channels=hidden_channel,
                                         k_size=kernel_size,
                                         stride=1,
                                         pad_size=(0, kernel_size[1] // 2),
                                         topology=topology,
                                         neighbor_dist=args["neighbor_dist_thresh"],
                                         ee_id=ee_id)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lkrelu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = SkeletonConvolution(in_channels=hidden_channel,
                                         out_channels=out_channel,
                                         k_size=kernel_size,
                                         stride=1,
                                         pad_size=(0, kernel_size[1] // 2),
                                         topology=topology,
                                         neighbor_dist=args["neighbor_dist_thresh"],
                                         ee_id=ee_id)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.lkrelu2 = nn.LeakyReLU(inplace=True)

        # self.pool = SkeletonPool(topology=topology, ee_id=ee_id, layer_idx=layer_idx)
        # self.new_topology = self.pool.new_topology
        # self.new_ee_id = self.pool.new_ee_id
        # this attribute is for Decoder to UpSampling
        # self.expand_num = self.pool.merge_nums

    def forward(self, x):
        identity = self.conv1by1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lkrelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.lkrelu2(out)
        # out = self.pool(out)
        return out


class StaticEncoder(nn.Module):
    """
    Encode the static offset with shape [B, 3, J, frame_num] into
    latent static tensor with shape [B, 32, J, frame_num]
    """

    def __init__(self, args, init_topology, init_ee_id):
        super(StaticEncoder, self).__init__()

        self.in_channel = 3
        self.out_channel = 16
        self.enc_layer = EncBasicBlock(args=args, in_channel=self.in_channel,
                                       out_channel=self.out_channel, topology=init_topology,
                                       ee_id=init_ee_id, layer_idx=1, dynamic=False)
        # init weights
        self.init_weights()

    def forward(self, x):
        out = self.enc_layer(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


class SkeletonEncoder(nn.Module):
    """
    The encoder which encodes the dynamic&static part of the animation data as mentioned in the paper.
    """

    def __init__(self, args, init_topology, init_ee_id):
        """
        :param args: options arguments
        :param init_topology: After parsing the bvh file, we'll get the init topology info of the simplified skeleton
        edges are a list obj which consists of many lists: [parent_node_idx, current_node_idx], it could tell us the
        init topology of the simplified skeleton
        :param init_ee_id: the end_effector index of the initial simplified skeleton
        """
        super(SkeletonEncoder, self).__init__()
        # store topologies for every SkeletonConvolution layer after SkeletonPooling
        self.topologies = [init_topology]
        self.ee_id_list = [init_ee_id]
        self.expand_num_list = []
        # self.in_channels = [7, 64]
        # self.out_channels = [32, 128]
        self.in_channels = [7, 32]
        self.out_channels = [16, 64]
        # build the 1st Encoder layer
        self.enc_layer1 = EncBasicBlock(args=args, in_channel=self.in_channels[0],
                                        out_channel=self.out_channels[0], topology=self.topologies[0],
                                        ee_id=self.ee_id_list[0], layer_idx=1)
        # self.topologies.append(self.enc_layer1.new_topology)
        # self.ee_id_list.append(self.enc_layer1.new_ee_id)
        # self.expand_num_list.append(self.enc_layer1.expand_num)
        # 2nd Encoder layer
        self.enc_layer2 = EncBasicBlock(args=args, in_channel=self.in_channels[1],
                                        out_channel=self.out_channels[1], topology=self.topologies[0],
                                        ee_id=self.ee_id_list[0], layer_idx=2)
        # self.expand_num_list.append(self.enc_layer2.expand_num)
        # init weights
        self.init_weights()

    def forward(self, x, s_latent):
        """
        :param x: The dynamic & static concatenate input[B, 7, joint_num, frame]
        :param s_latent: The static latent feature input[B, 32, joint_num, frame]
        :return: tensor with shape [B, 128, joint_num_after_2_pooling, frame]
        """
        # [B, 7, joint_num, frame] -> [B, 32, joint_num, frame]
        out = self.enc_layer1(x)
        # [B, 32, joint_num, frame] -> [B, 64, joint_num, frame]
        out = torch.cat([out, s_latent], dim=1)
        # [B, 64, joint_num, frame] -> [B, 128, joint_num, frame]
        out = self.enc_layer2(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
