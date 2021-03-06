import torch.utils
import torch.utils.cpp_extension

from model.utils.functions import *
from model.network import *


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def mix_reparameterize(mu, logvar, t_mu, t_logvar, ratio):
    mix_mean = ratio * t_mu + (1 - ratio) * mu
    mix_var = (1 - ratio) * logvar + ratio * t_logvar
    # mix_mean = ratio/(1+ratio) * t_mu + 1/(ratio+1) * mu
    # mix_var = ratio/(1+ratio) * t_logvar + 1/(ratio+1) * logvar
    std = torch.exp(0.5 * mix_var)
    eps = torch.randn_like(std)
    return mix_mean + eps * std, mix_mean, mix_var


class CVAE(nn.Module):
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
        self.d_lstm = LSTM([self.model_conf["latent_size"]
                            + self.model_conf["encoder"]["target_dim"][-1]
                            + self.model_conf["encoder"]["state_dim"][-1],  # 16 + 256 + 256
                            self.model_conf["lstm"]["lstm_dim"][-1]], self.model_conf["lstm"]["layer_num"]).cuda()
        # self.d_lstm = LSTM([self.model_conf["latent_size"] + self.model_conf["encoder"]["target_dim"][-1],  # 16 + 256
        #                     self.model_conf["lstm"]["lstm_dim"][-1]], self.model_conf["lstm"]["layer_num"]).cuda()

        # mean & var
        self.l_mn = torch.nn.Linear(self.model_conf["lstm"]["lstm_dim"][-1], self.model_conf["latent_size"]).cuda()
        self.l_var = torch.nn.Linear(self.model_conf["lstm"]["lstm_dim"][-1], self.model_conf["latent_size"]).cuda()

        # target mean & var
        self.l_t_mn = torch.nn.Linear(self.model_conf["lstm"]["lstm_dim"][-1], self.model_conf["latent_size"]).cuda()
        self.l_t_var = torch.nn.Linear(self.model_conf["lstm"]["lstm_dim"][-1], self.model_conf["latent_size"]).cuda()

        # build decoder network
        self.decoder = Decoder(self.model_conf["decoder"]["decoder_dim"], self.model_conf["decoder"]["out_dim"],
                               self.model_conf["decoder"]["contact_dim"]).cuda()

    def sample(self, batch_size):
        sampled_embedding = np.random.randn(1,
                                            batch_size,
                                            self.model_conf["latent_size"])

        return sampled_embedding

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
        self.d_lstm.init_hidden(len(batch_sample["local_quaternion"]))

        root_loss = 0
        contact_loss = 0
        position_loss = 0
        quaternion_loss = 0
        KLD = 0

        pred_list = [global_position[:, 0]]  # B*22*3
        contact_list = [contact[:, 0]]
        pred_quat_list = [local_quaternion[:, 0].view(local_quaternion[:, 0].size(0), -1)]
        bvh_list = [torch.cat([global_position[:, 0, 0],
                               local_quaternion[:, 0, ].view(local_quaternion.size(0), -1)],
                              -1)]

        # ttime to arrival: 0
        target_contact = contact[:, -1]
        target_local_quaternion = local_quaternion[:, -1]
        target_local_quaternion = target_local_quaternion.view(target_local_quaternion.size(0), -1)
        target_root_velocity = root_velocity[:, -1]

        target_state = torch.cat([target_local_quaternion, target_root_velocity, target_contact], -1)

        for t in range(cur_sequence_length - 1):
            if t == 0:
                # state
                cur_contact = contact[:, t]
                cur_local_quaternion = local_quaternion[:, t]
                cur_local_quaternion = cur_local_quaternion.view(cur_local_quaternion.size(0), -1)
                cur_root_velocity = root_velocity[:, t]
                cur_root_position = root_position[:, t]
            else:
                # state
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

            target_state_embedding = self.state_encoder(target_state)
            target_before_embedding = target_embedding

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

            # target - concat separate embeddings
            target_concat_embedding = torch.cat([target_state_embedding,
                                                 torch.zeros(tuple(offset_embedding.shape)).cuda(),
                                                 target_before_embedding], -1).unsqueeze(0)
            hidden_target_state = self.lstm(target_concat_embedding)
            t_mean, t_var = self.l_t_mn(hidden_target_state), self.l_t_var(hidden_target_state)

            ratio = gen_ratio(t)

            # Train
            if train:
                # lstm
                hidden_state = self.lstm(input_embedding)
                # cal distribution
                mean, var = self.l_mn(hidden_state), self.l_var(hidden_state)
                # sample
                # sampled_embedding = reparameterize(mean, var)
                sampled_embedding, mean, var = mix_reparameterize(mean, var, t_mean, t_var, ratio)

            # Inference
            else:
                # sampled_embedding = self.sample(len(batch_sample["local_quaternion"]))
                # sampled_embedding = torch.FloatTensor(sampled_embedding).cuda()

                hidden_state = self.lstm(input_embedding)
                mean, var = self.l_mn(hidden_state), self.l_var(hidden_state)
                sampled_embedding, mean, var = mix_reparameterize(mean, var, t_mean, t_var, ratio)

            # z, target embedding and state embedding
            concat_sampled_embedding = torch.cat([sampled_embedding,
                                                  target_embedding.unsqueeze(0),
                                                  state_embedding.unsqueeze(0)]
                                                 , -1)

            # decode
            hidden_state = self.d_lstm(concat_sampled_embedding)
            output, pred_contact = self.decoder(hidden_state)

            # update quaternion
            pred_local_quaternion = cur_local_quaternion + output[:, :, :self.data_conf["target_input_dim"]]
            pred_local_quaternion_norm = pred_local_quaternion.view(pred_local_quaternion.size(0),
                                                                    pred_local_quaternion.size(1), -1, 4)
            pred_local_quaternion_norm = pred_local_quaternion_norm / torch.norm(pred_local_quaternion_norm,
                                                                                 dim=-1,
                                                                                 keepdim=True)
            pred_quat_list.append(pred_local_quaternion[0])

            # update root
            pred_root_velocity = output[:, :, self.data_conf["target_input_dim"]:]
            pred_root_position = cur_root_position + pred_root_velocity
            pred_global_position = self.skeleton.forward_kinematics(pred_local_quaternion_norm,
                                                                    pred_root_position)
            pred_list.append(pred_global_position[0])
            contact_list.append(pred_contact[0])
            # pred_list.append(global_position[:, t+1])
            # contact_list.append(contact[:, t+1])

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

            if train:
                # KLD
                # var = torch.pow(torch.exp(var), 2)
                # KLD += -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var) / cur_sequence_length

                # robust KLD
                KLD += -0.5 * torch.mean(1 + var - mean.pow(2) - var.exp()) / cur_sequence_length

                # normal KLD
                # KLD += -0.5 * torch.sum(1 + var - torch.pow(mean, 2) - var.exp()) / cur_sequence_length

        # preprocess contact
        pred_contact_list = torch.cat(contact_list, 0)
        pred_contact_list = pred_contact_list.view(len(contact_list), -1, 4).permute(1, 0, 2)  # BxTx4
        pred_contact_list[pred_contact_list > 0.5] = 1
        pred_contact_list[pred_contact_list <= 0.5] = 0

        if train:
            all_pred = [pred_list, pred_contact_list]
            all_loss = [root_loss, contact_loss, quaternion_loss, position_loss, KLD]
        else:
            # all_pred = [pred_list, pred_quat_list, bvh_list, pred_img_list, gt_img_list, img_list]
            all_pred = [pred_list, pred_contact_list, contact_list, bvh_list]
            all_loss = [root_loss, contact_loss, quaternion_loss, position_loss]

        return all_pred, all_loss
