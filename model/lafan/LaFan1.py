from torch.utils.data.dataset import T_co

from model.lafan import extract, utils
import numpy as np
from torch.utils.data import Dataset


class LaFan1(Dataset):
    """
    A very basic animation object
    """
    def __init__(self, bvh_path, train=False, window=50, offset=20, flip=False):
        if train:
            # self.actors = ["subject1", "subject2", "subject3", "subject4"]
            self.actors = ["subject6"]
        else:
            # self.actors = ["subject5"]
            self.actors = ["subject6"]
        self.window = window
        self.offset = offset
        self.offsets = None
        self.parents = None
        self.p_std = 1
        self.v_std = 1
        self.flip = flip
        self.start_sequence_length = 5
        self.cur_sequence_length = 5

        label = "train" if train else "test"
        print("Building the " + label + " set...")
        if flip:
            self.data, self.iv_data = self.load_data(bvh_path)
        else:
            self.data = self.load_data(bvh_path)

    def load_data(self, bvh_path):
        # X: local positions
        # Q: local quaternions
        # parents: list of parent indices defining the bone hierarchy
        # contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        # contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
        X, Q, self.parents, contacts_l, contacts_r = \
            extract.get_lafan1_set(bvh_path, self.actors, self.window, self.offset)

        # global quaternion, global position n x sequence_length x joint_num x 4/3
        global_quaternion, global_position = utils.quat_fk(Q, X, self.parents)

        # global positions stats
        self.p_std = np.std(
            global_position.reshape([global_position.shape[0], global_position.shape[1], -1]).transpose([0, 2, 1]),
            axis=(0, 2),
            keepdims=True)

        # global velocity std
        gp = global_position.reshape(global_position.shape[0], global_position.shape[1], -1)
        all_position = []
        all_position.extend(gp[0, :])
        for i in range(gp.shape[0] - 1):
            all_position.extend(gp[i + 1, -20:])
        all_position = np.asarray(all_position)
        all_velocity = all_position[1:] - all_position[:-1]
        self.v_std = np.std(all_velocity, axis=0)

        data = {
            # 1. local quaternion vector (J * 4d)
            "local_quaternion": Q,
            # 2. global root velocity vector (3d)
            "root_velocity": global_position[:, 1:, 0, :] - global_position[:, :-1, 0, :],
            # 3. contact information vector (4d)
            "contact": np.concatenate([contacts_l, contacts_r], -1),
            # 4. global root position offset (?d)
            # last frame root position
            "root_position_offset": global_position[:, -1, 0, :],
            # 5. local quaternion offset (?d)
            # last frame all quaternion
            "local_quaternion_offset": Q[:, -1, :, :],
            # 6. target
            # last frame (quaternion)
            "target": Q[:, -1, :, :],
            # 7. root pos
            "root_position": global_position[:, :, 0, :],
            # 8. global_position
            "global_position": global_position[:, :, :, :],
            # 9. global quaternion
            # "global_quaternion": global_quaternion[:, :, :, :]
        }

        # flip temporal
        if self.flip:
            Q = Q[:, ::-1, :, :]
            contacts_l = contacts_l[:, ::-1, :]
            contacts_r = contacts_r[:, ::-1, :]
            global_position = global_position[:, ::-1, :, :]

            iv_data = {
                # 1. local quaternion vector (J * 4d)
                "local_quaternion": Q,
                # 2. global root velocity vector (3d)
                "root_velocity": global_position[:, 1:, 0, :] - global_position[:, :-1, 0, :],
                # 3. contact information vector (4d)
                "contact": np.concatenate([contacts_l, contacts_r], -1),
                # 4. global root position offset (?d)
                # last frame root position
                "root_position_offset": global_position[:, -1, 0, :],
                # 5. local quaternion offset (?d)
                # last frame all quaternion
                "local_quaternion_offset": Q[:, -1, :, :],
                # 6. target
                # last frame (quaternion)
                "target": Q[:, -1, :, :],
                # 7. root pos
                "root_position": global_position[:, :, 0, :],
                # 8. global_position
                "global_position": global_position[:, :, :, :],
                # 9. global quaternion
                # "global_quaternion": global_quaternion[:, :, :, :]
            }
            return data, iv_data
        return data

    def __len__(self):
        return len(self.data["local_quaternion"])

    def __getitem__(self, index) -> T_co:
        sample = {"local_quaternion": self.data["local_quaternion"][index].astype(np.float32),
                  "root_velocity": self.data["root_velocity"][index].astype(np.float32),
                  "contact": self.data["contact"][index].astype(np.float32),
                  "root_position_offset": self.data["root_position_offset"][index].astype(np.float32),
                  "local_quaternion_offset": self.data["local_quaternion_offset"][index].astype(np.float32),
                  "target": self.data["target"][index].astype(np.float32),
                  "root_position": self.data["root_position"][index].astype(np.float32),
                  "global_position": self.data["global_position"][index].astype(np.float32),
                  # "global_quaternion": self.data["global_quaternion"][index].astype(np.float32),
                  }

        if self.flip:
            iv_sample = {"local_quaternion": self.iv_data["local_quaternion"][index].astype(np.float32),
                         "root_velocity": self.iv_data["root_velocity"][index].astype(np.float32),
                         "contact": self.iv_data["contact"][index].astype(np.float32),
                         "root_position_offset": self.iv_data["root_position_offset"][index].astype(np.float32),
                         "local_quaternion_offset": self.iv_data["local_quaternion_offset"][index].astype(np.float32),
                         "target": self.iv_data["target"][index].astype(np.float32),
                         "root_position": self.iv_data["root_position"][index].astype(np.float32),
                         "global_position": self.iv_data["global_position"][index].astype(np.float32),
                         # "global_quaternion": self.iv_data["global_quaternion"][index].astype(np.float32),
                         }
            return sample, iv_sample
        return sample


if __name__ == "__main__":
    lafan = LaFan1("C:/Users/alantxren/ubisoft-laforge-animation-dataset/lafan1/lafan1_bvh", False)
    for key in lafan.data.keys():
        print(key)
        print(lafan.data[key].shape)
