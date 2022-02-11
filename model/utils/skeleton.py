import torch
import numpy as np
from model.utils.quanternion import qmul, qrot, qinv


class Skeleton:
    def __init__(self, offsets, parents, joints_left=None, joints_right=None):
        assert len(offsets) == len(parents)

        self._offsets = torch.FloatTensor(offsets)
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()
        if torch.cuda.is_available():
            self.cuda()

    def cuda(self):
        self._offsets = self._offsets.cuda()
        return self

    def num_joints(self):
        return self._offsets.shape[0]

    def offsets(self):
        return self._offsets

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def end_joint(self):
        return np.where(~self._has_children)[0]

    def key_joint(self):
        return np.where(self._has_children)[0]

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove, dataset=None):
        """
        Remove the joints specified in 'joints_to_remove', both from the
        skeleton definition and from the dataset (which is modified in place).
        The rotations of removed joints are propagated along the kinematic chain.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        # # Update all transformations in the dataset
        # for subject in dataset.subjects():
        #     for action in dataset[subject].keys():
        #         rotations = dataset[subject][action]['rotations']
        #         for joint in joints_to_remove:
        #             for child in self._children[joint]:
        #                 rotations[:, child] = qmul(rotations[:, joint], rotations[:, child])
        #             rotations[:, joint] = [1, 0, 0, 0]  # Identity
        #         dataset[subject][action]['rotations'] = rotations[:, valid_joints]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        self._offsets = self._offsets[valid_joints]
        self._compute_metadata()

    def forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (B, L, J, 4) or (B, J, 4) tensor of unit quaternions describing the local rotation of each joint.
         -- root_positions: (B, L, 3) or (B, 3) tensor describing the root joint positions.
        """
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(*rotations.shape[:-2], self._offsets.shape[0], self._offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[..., 0, :])
            else:
                positions_world.append(
                    qrot(rotations_world[self._parents[i]], expanded_offsets[..., i, :]) + positions_world[
                        self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(qmul(rotations_world[self._parents[i]], rotations[..., i, :]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=-2)  # stack along the joints axis

    def inverse_kinematics(self, rotation, position):
        assert rotation.shape[-1] == 4
        assert position.shape[-1] == 3

        # local_rotation = torch.cat(
        #     [rotation[..., :1, :], qmul(qinv(rotation[..., self._parents[1:], :]), rotation[..., 1:, :])], dim=-2)
        local_position = torch.cat([position[..., :1, :],
                                    qrot(qinv(rotation[..., self._parents[1:], :]),
                                         position[..., 1:, :] - position[..., self._parents[1:], :])], dim=-2)
        return local_position

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        # for i, parent in enumerate(self._parents):
        #     self._children.append([])
        for i, parent in enumerate(self._parents):
            self._children.append([])
            if parent != -1:
                self._children[parent].append(i)


if __name__ == "__main__":
    offsets = [[0.0, 0.0, 0.0], [1.035e-01, 1.858e+00, 1.055e+01], [4.350e+01, -2.700e-05, -2.000e-06],
               [4.237e+01, -1.100e-05, -1.000e-05], [1.730e+01, 1.000e-06, 4.000e-06],
               [1.035e-01, 1.858e+00, -1.055e+01], [4.350e+01, -3.100e-05, 1.500e-05],
               [4.237e+01, -1.900e-05, 1.000e-05], [1.730e+01, -4.000e-06, 6.000e-06],
               [6.902e+00, -2.604e+00, 6.000e-06], [1.259e+01, -2.000e-06, 2.000e-06],
               [1.234e+01, 1.300e-05, -1.400e-05], [2.583e+01, -1.700e-05, -2.000e-06],
               [1.177e+01, 1.900e-05, 5.000e-06], [1.975e+01, -1.480e+00, 6.000e+00],
               [1.128e+01, 4.000e-06, -2.800e-05], [3.300e+01, 6.000e-06, 3.300e-05],
               [2.520e+01, -7.000e-06, -7.000e-06], [1.975e+01, -1.480e+00, -6.000e+00],
               [1.128e+01, -2.900e-05, -2.300e-05], [3.300e+01, 1.300e-05, 1.000e-05],
               [2.520e+01, 1.620e-04, 4.380e-04]]
    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

    skeleton = Skeleton(offsets, parents)
    print(skeleton.has_children())
