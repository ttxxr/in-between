import torch.utils
import torch.utils.cpp_extension

from PIL import Image
from model.network import *
from model.utils.functions import *


class CGAN(nn.Module):
    def __init__(self, opt, skeleton):
        super().__init__()
        self.data_conf = opt["data"]
        self.model_conf = opt["model"]
        self.skeleton = skeleton

        # build separate encoder network
        # separate: state, offset, target
        self.state_encoder = Encoder(self.model_conf["encoder"]["state_dim"], dropout=0).cuda()
        self.offset_encoder = Encoder(self.model_conf["encoder"]["offset_dim"], dropout=0).cuda()
        self.target_encoder = Encoder(self.model_conf["encoder"]["target_dim"], dropout=0).cuda()

        # build lstm network
        self.lstm = LSTM(self.model_conf["lstm"]["lstm_dim"], self.model_conf["lstm"]["layer_num"]).cuda()

        # build decoder network
        self.decoder = Decoder(self.model_conf["decoder"]["decoder_dim"],
                               self.model_conf["decoder"]["out_dim"],
                               self.model_conf["decoder"]["contact_dim"]).cuda()

    def forward(self, batch_sample, z_tta, std, cur_sequence_length, train=True):
        # state info
        local_quaternion = batch_sample["local_quaternion"].cuda()
        root_velocity = batch_sample["root_velocity"].cuda()
        contact = batch_sample["contact"].cuda()
        # offset info
        root_position_offset = batch_sample["root_position_offset"].cuda()
        local_quaternion_offset = batch_sample["local_quaternion_offset"].cuda()
        local_quaternion_offset = local_quaternion_offset.view(local_quaternion_offset.size(0), -1)
        # target
        target = batch_sample["target"].cuda()
        target = target.view(target.size(0), -1)
        # root position
        root_position = batch_sample["root_position"].cuda()
        # global_position
        global_position = batch_sample["global_position"].cuda()

        # init lstm
        self.lstm.init_hidden(len(batch_sample["local_quaternion"]))

        root_loss = 0
        contact_loss = 0
        position_loss = 0
        quaternion_loss = 0
        pred_list = [global_position[:, 0]]  # B*22*3
        bvh_list = [torch.cat([global_position[:, 0, 0],
                               local_quaternion[:, 0, ].view(local_quaternion.size(0), -1)],
                              -1)]

        img_list = []
        gt_img_list = []
        pred_img_list = []

        for t in range(cur_sequence_length - 1):
            # state
            if t == 0:
                cur_contact = contact[:, t]
                cur_local_quaternion = local_quaternion[:, t]
                cur_local_quaternion = cur_local_quaternion.view(cur_local_quaternion.size(0), -1)
                cur_root_velocity = root_velocity[:, t]
                cur_root_position = root_position[:, t]
            else:
                cur_contact = pred_contact[0]
                cur_local_quaternion = pred_local_quaternion[0]
                cur_root_velocity = pred_root_velocity[0]
                cur_root_position = pred_root_position[0]

            # offset
            cur_root_position_offset = root_position_offset - cur_root_position
            cur_local_quaternion_offset = local_quaternion_offset - cur_local_quaternion

            state_input = torch.cat([cur_local_quaternion, cur_root_velocity, cur_contact], -1)
            offset_input = torch.cat([cur_root_position_offset, cur_local_quaternion_offset], -1)
            target_input = target

            # encode
            state_embedding = self.state_encoder(state_input)
            offset_embedding = self.offset_encoder(offset_input)
            target_embedding = self.target_encoder(target_input)

            # add Ztta
            state_embedding += z_tta[:, t + 1]
            offset_embedding += z_tta[:, t + 1]
            target_embedding += z_tta[:, t + 1]
            robust_embedding = torch.cat([offset_embedding, target_embedding], -1)

            # generate target noise embedding
            lambda_target = gen_ztarget(t, cur_sequence_length)
            z_target = self.model_conf["noise_theta"] * lambda_target * torch.FloatTensor(
                robust_embedding.size()).normal_().cuda()
            robust_embedding += z_target

            # concat separate embeddings
            input_embedding = torch.cat([state_embedding, robust_embedding], -1).unsqueeze(0)

            # lstm
            hidden_state = self.lstm(input_embedding)

            # decode
            output, pred_contact = self.decoder(hidden_state)

            # update quaternion
            pred_local_quaternion = cur_local_quaternion + output[:, :, :self.data_conf["target_input_dim"]]
            pred_local_quaternion_norm = pred_local_quaternion.view(pred_local_quaternion.size(0),
                                                                    pred_local_quaternion.size(1), -1, 4)
            pred_local_quaternion_norm = pred_local_quaternion_norm / torch.norm(pred_local_quaternion_norm, dim=-1,
                                                                                 keepdim=True)

            # update root
            pred_root_velocity = output[:, :, self.data_conf["target_input_dim"]:]
            pred_root_position = cur_root_position + pred_root_velocity
            pred_global_position = self.skeleton.forward_kinematics(pred_local_quaternion_norm,
                                                                    pred_root_position)
            pred_list.append(pred_global_position[0])

            # bvh
            bvh_list.append(torch.cat([pred_root_position[0], pred_local_quaternion[0]], -1))

            # quaternion loss
            actual_local_quaternion = local_quaternion[:, t + 1].view(local_quaternion[:, t + 1].size(0), -1)
            quaternion_loss += torch.mean(torch.abs(
                pred_local_quaternion[0] - actual_local_quaternion)) / cur_sequence_length
            # root loss
            root_loss += torch.mean(torch.abs(pred_root_position[0] - root_position[:, t + 1])
                                    / std[:, :, 0]) / cur_sequence_length
            # contact loss
            contact_loss += torch.mean(
                torch.abs(pred_contact[0] - contact[:, t + 1])) / cur_sequence_length
            # global position loss
            position_loss += torch.mean(torch.abs(
                pred_global_position[0] - global_position[:, t + 1]) / std) / cur_sequence_length

        all_loss = [root_loss, contact_loss, quaternion_loss, position_loss]
        if train:
            all_pred = [pred_list]
        else:
            all_pred = [pred_list, bvh_list]

        return all_pred, all_loss
