import logging
import os

import imageio
import matplotlib
import torch.optim as optim
import torch.utils
import torch.utils.cpp_extension
import torch.utils.data as tordata
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.lafan.benchmarks import fast_npss
from model.network.cvae_gan.cvae_bi import *
from model.utils.metrics import *

matplotlib.use("Agg")


def blend(normal_gp, inverse_gp):
    blend_list = []
    blend_left_border = 5
    blend_right_border = 45
    for i in range(len(normal_gp)):
        if i < blend_left_border:
            weight = 1
        elif i >= blend_right_border:
            weight = 0
        else:
            weight = 1 - (i - blend_left_border) / blend_right_border
        blend_list.append(weight * normal_gp[i] + (1 - weight) * inverse_gp[len(normal_gp) - i - 1])
    return blend_list


class Model(object):
    def __init__(self, opt, skeleton, train_data=None, test_data=None):
        self.data_conf = opt["data"]
        self.model_conf = opt["model"]
        self.train_conf = opt["network"]

        self.lr = self.train_conf["learning_rate"]
        self.epoch_num = self.train_conf["epoch"]
        self.batch_size = self.train_conf["batch_size"]
        self.save_path = self.train_conf["save_path"]
        self.load_path = self.train_conf["load_path"]

        self.train_data = train_data
        self.test_data = test_data
        self.cur_sequence_length = self.test_data.cur_sequence_length
        self.p_std = None
        self.v_std = None

        # build cvae
        self.cvae = CVAE(opt, skeleton)
        self.iv_cvae = CVAE(opt, skeleton)

        # build discriminator network
        self.long_discriminator = LongMotionDiscriminator(length=self.data_conf["short_length"],
                                                          in_dim=self.data_conf["joint_num"] * 3 * 2).cuda()
        self.short_discriminator = ShortMotionDiscriminator(length=self.data_conf["short_length"],
                                                            in_dim=self.data_conf["joint_num"] * 3 * 2).cuda()
        # TODO: discriminator before blend
        self.positive_discriminator = LongMotionDiscriminator(length=self.data_conf["short_length"],
                                                              in_dim=self.data_conf["joint_num"] * 3 * 2).cuda()

        # build optimizer
        self.optimizer = optim.Adam(lr=self.lr,
                                    params=(list(self.cvae.parameters()) +
                                            list(self.iv_cvae.parameters())),
                                    betas=(self.train_conf["beta1"], self.train_conf["beta2"]),
                                    weight_decay=self.train_conf["weight_decay"])
        self.optimizer_discriminator = optim.Adam(lr=self.lr * 0.1,
                                                  params=(list(self.long_discriminator.parameters()) +
                                                          list(self.short_discriminator.parameters()) +
                                                          list(self.positive_discriminator.parameters())),
                                                  betas=(self.train_conf["beta1"], self.train_conf["beta2"]),
                                                  weight_decay=self.train_conf["weight_decay"])

        # self.loss_function = nn.MSELoss(reduction='mean')

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

    def load(self, epoch=0):
        epoch = "" if epoch == 0 else str(epoch)

        self.cvae.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "cvae.pth")))
        self.iv_cvae.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "iv_cvae.pth")))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.train_conf["load_path"], epoch + "optimizer.pth")))

        # discriminator
        self.short_discriminator.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "short_discriminator.pth")))
        self.long_discriminator.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "long_discriminator.pth")))
        self.positive_discriminator.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "positive_discriminator.pth")))
        self.optimizer_discriminator.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "optimizer_d.pth")))

    def save(self, epoch=""):
        torch.save(self.cvae.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "cvae.pth"))
        torch.save(self.iv_cvae.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "iv_cvae.pth"))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "optimizer.pth"))

        # discriminator
        torch.save(self.short_discriminator.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "short_discriminator.pth"))
        torch.save(self.long_discriminator.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "long_discriminator.pth"))
        torch.save(self.positive_discriminator.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "positive_discriminator.pth"))
        torch.save(self.optimizer_discriminator.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "optimizer_d.pth"))

    def train(self):
        writer = SummaryWriter(self.train_conf["log_path"])
        train_loader = tordata.DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=4,
                                          shuffle=True)
        # save global position std
        std = np.concatenate((self.train_data.p_std[0], self.train_data.v_std.reshape((66, 1))), axis=1)
        np.savetxt(os.path.join(self.save_path, "norm.csv"), std)
        self.p_std = torch.from_numpy(self.train_data.p_std).view(1, 1, -1, 3).cuda()
        self.v_std = torch.from_numpy(self.train_data.v_std).view(1, 1, -1, 3).cuda()

        min_loss = 1000000
        for epoch in range(self.epoch_num):
            self.cvae.train()
            self.iv_cvae.train()

            # update current sequence length
            # TODO: update progressive learning
            temp_len = self.train_data.start_sequence_length + int(epoch / self.data_conf["seq_len_update_rate"])
            self.cur_sequence_length = temp_len if temp_len < self.data_conf["sequence_length"] \
                else self.data_conf["sequence_length"]

            # Ztta embedding
            z_tta = gen_ztta(self.cur_sequence_length, self.model_conf["encoder"]["state_dim"][-1],
                             self.model_conf["position_encoding_basis"]).cuda()

            batch_idx = 0
            batch_loss = []
            for two_samples in tqdm(train_loader, ncols=100):
                batch_sample, iv_batch_sample = two_samples
                # forward
                all_pred, all_loss = self.cvae(batch_sample, z_tta, self.p_std, self.cur_sequence_length)
                iv_all_pred, iv_all_loss = self.iv_cvae(iv_batch_sample, z_tta, self.p_std, self.cur_sequence_length)

                pred_list = all_pred[0]
                root_loss, contact_loss, quaternion_loss, position_loss, KLD = all_loss

                iv_pred_list = iv_all_pred[0]
                iv_root_loss, iv_contact_loss, iv_quaternion_loss, iv_position_loss, iv_KLD = iv_all_loss

                blend_list = blend(pred_list, iv_pred_list)

                # adversarial input before blend
                positive_fake_position = torch.cat([item.reshape(item.size(0), -1).unsqueeze(-1) for item in pred_list],
                                                   -1)
                positive_fake_velocity = torch.cat(
                    [positive_fake_position[:, :, 1:] - positive_fake_position[:, :, :-1],
                     torch.zeros_like(positive_fake_position[:, :, 0:1])], -1)
                positive_fake_input = torch.cat([positive_fake_position, positive_fake_velocity], 1)

                positive_real_position = torch.cat(
                    [batch_sample["global_position"][:, i].view(batch_sample["global_position"].size(0), -1).unsqueeze(
                        -1) for i in range(self.cur_sequence_length)], -1)
                positive_real_velocity = torch.cat(
                    [positive_real_position[:, :, 1:] - positive_real_position[:, :, :-1],
                     torch.zeros_like(positive_real_position[:, :, 0:1])], -1)
                positive_real_input = torch.cat([positive_real_position, positive_real_velocity], 1)

                # adversarial input
                fake_position = torch.cat([item.reshape(item.size(0), -1).unsqueeze(-1) for item in blend_list], -1)
                fake_velocity = torch.cat([fake_position[:, :, 1:] - fake_position[:, :, :-1],
                                           torch.zeros_like(fake_position[:, :, 0:1])], -1)
                fake_input = torch.cat([fake_position, fake_velocity], 1)

                real_position = torch.cat(
                    [batch_sample["global_position"][:, i].view(batch_sample["global_position"].size(0), -1).unsqueeze(
                        -1) for i in range(self.cur_sequence_length)], -1)
                real_velocity = torch.cat([real_position[:, :, 1:] - real_position[:, :, :-1],
                                           torch.zeros_like(real_position[:, :, 0:1])], -1)
                real_input = torch.cat([real_position, real_velocity], 1)

                # discriminator
                self.optimizer_discriminator.zero_grad()
                short_fake_logic = torch.mean(self.short_discriminator(fake_input.detach())[:, 0], 1)
                short_real_logic = torch.mean(self.short_discriminator(real_input.cuda())[:, 0], 1)
                short_fake_loss = torch.mean(short_fake_logic ** 2)
                short_real_loss = torch.mean((short_real_logic - 1) ** 2)
                short_d_loss = (short_real_loss + short_fake_loss) / 2

                long_fake_logic = torch.mean(self.long_discriminator(fake_input.detach())[:, 0], 1)
                long_real_logic = torch.mean(self.long_discriminator(real_input.cuda())[:, 0], 1)
                long_fake_loss = torch.mean(long_fake_logic ** 2)
                long_real_loss = torch.mean((long_real_logic - 1) ** 2)
                long_d_loss = (long_real_loss + long_fake_loss) / 2

                positive_fake_logic = torch.mean(self.positive_discriminator(positive_fake_input.detach())[:, 0], 1)
                positive_real_logic = torch.mean(self.positive_discriminator(positive_real_input.cuda())[:, 0], 1)
                positive_fake_loss = torch.mean(positive_fake_logic ** 2)
                positive_real_loss = torch.mean((positive_real_logic - 1) ** 2)
                positive_d_loss = (positive_real_loss + positive_fake_loss) / 2

                discriminator_loss = self.train_conf["gan_loss_weight"] * (short_d_loss + long_d_loss + positive_d_loss)
                discriminator_loss.backward()
                self.optimizer_discriminator.step()

                # generator
                self.optimizer.zero_grad()
                short_g_fake_logic = torch.mean(self.short_discriminator(fake_input)[:, 0], 1)
                short_g_loss = torch.mean((short_g_fake_logic - 1) ** 2)
                long_g_fake_logic = torch.mean(self.long_discriminator(fake_input)[:, 0], 1)
                long_g_loss = torch.mean((long_g_fake_logic - 1) ** 2)
                positive_g_fake_logic = torch.mean(self.positive_discriminator(positive_fake_input)[:, 0], 1)
                positive_g_loss = torch.mean((positive_g_fake_logic - 1) ** 2)

                # robust KLD
                KLD = torch.sqrt(1 + KLD ** 2) - 1

                # foot sliding
                pred_pos = torch.cat([x.reshape(x.size(0), -1).unsqueeze(-1) for x in blend_list], -1)
                pred_vel = (pred_pos[:, self.data_conf["foot_index"], 1:] -
                            pred_pos[:, self.data_conf["foot_index"], :-1])
                pred_vel = pred_vel.view(pred_vel.size(0), 4, 3, pred_vel.size(-1))
                # B*4[foot joint num]*3*T[49]  contact:B*T*4(B*4*T[49]*1)
                slide_loss = torch.mean(torch.abs(pred_vel *
                                                  batch_sample["contact"][:, :self.cur_sequence_length - 1]
                                                  .permute(0, 2, 1).unsqueeze(2).cuda()))

                # sum loss
                total_loss = (self.train_conf["quaternion_loss_weight"] * (quaternion_loss + iv_quaternion_loss) / 2 +
                              self.train_conf["contact_loss_weight"] * (contact_loss + iv_contact_loss) / 2 +
                              self.train_conf["root_loss_weight"] * (root_loss + iv_root_loss) / 2 +
                              self.train_conf["position_loss_weight"] * (position_loss + iv_position_loss) / 2 +
                              self.train_conf["gan_loss_weight"] * (short_g_loss + long_g_loss + positive_g_loss) +
                              self.train_conf["kld_weight"] * (KLD + iv_KLD) / 2 +
                              self.train_conf["slide_loss_weight"] * slide_loss)
                total_loss.backward()

                # gradient clip
                nn.utils.clip_grad_norm_(self.cvae.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.iv_cvae.parameters(), 0.5)

                self.optimizer.step()

                # generator loss
                writer.add_scalar("root_loss", root_loss + iv_root_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("contact_loss", contact_loss + iv_contact_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("position_loss", position_loss + iv_position_loss,
                                  global_step=epoch * 317 + batch_idx)
                writer.add_scalar("quaternion_loss", quaternion_loss + iv_quaternion_loss,
                                  global_step=epoch * 317 + batch_idx)
                writer.add_scalar("KLD", KLD + iv_KLD, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("slide_loss", slide_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("total_generator_loss", total_loss, global_step=epoch * 317 + batch_idx)

                # GAN loss
                writer.add_scalar("short_generator_loss", short_g_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("long_generator_loss", long_g_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("positive_generator_loss", positive_g_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("short_discriminator_loss", short_d_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("long_discriminator_loss", long_d_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("positive_discriminator_loss", positive_d_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("discriminator_loss", discriminator_loss, global_step=epoch * 317 + batch_idx)

                batch_loss.append(total_loss.cpu().detach())
                batch_idx += 1

            # save model
            cur_train_loss = np.mean(batch_loss)
            if cur_train_loss < min_loss:
                min_loss = cur_train_loss
                self.save()

            if epoch % 5 == 0:
                test_loss = self.test()
                writer.add_scalar("test loss", test_loss, global_step=epoch * 317)

            if epoch % self.train_conf["save_duration"] == 0 and epoch > 135:
                self.save(epoch.__str__())

            logging.info('Epoch {} : '.format(epoch) +
                         'Train Loss = {:.9f} '.format(cur_train_loss) +
                         'Min Loss = {:.9f} '.format(min_loss) +
                         'lr = {} '.format(self.lr))

            print('Epoch {} : '.format(epoch) +
                  'Train Loss = {:.9f} '.format(cur_train_loss) +
                  'Min Loss = {:.9f} '.format(min_loss) +
                  'lr = {} '.format(self.lr))

        writer.close()

    def test(self):
        test_loader = tordata.DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=4,
                                         shuffle=False)

        # load global position std
        std = np.loadtxt(os.path.join(self.load_path, "norm.csv"))
        self.p_std = torch.from_numpy(std[:, 0]).view(1, 1, -1, 3).cuda()
        # self.v_std = torch.from_numpy(std[:, 1]).view(1, 1, -1, 3).cuda()

        # load old norm (12-01)
        # self.p_std = torch.from_numpy(np.loadtxt(os.path.join(self.load_path, "norm.csv"))).view(1, 1, -1, 3).cuda()
        # self.v_std = self.p_std

        self.cvae.eval()
        self.iv_cvae.eval()
        self.cur_sequence_length = self.data_conf["sequence_length"]

        # Ztta embedding
        z_tta = gen_ztta(self.cur_sequence_length, self.model_conf["encoder"]["state_dim"][-1],
                         self.model_conf["position_encoding_basis"]).cuda()

        metrics = []
        batch_loss = []
        for batch_idx, two_samples in enumerate(test_loader):
            batch_sample, iv_batch_sample = two_samples

            # select sample
            # if batch_idx < 98:
            #     continue
            if self.data_conf["test_nums"] == "single" and batch_idx not in self.data_conf["single_idx"]:
                continue
            if self.data_conf["save_gif"]:
                print(batch_idx, " ", batch_sample["global_position"].shape)

            with torch.no_grad():
                all_pred, all_loss = self.cvae(batch_sample, z_tta, self.p_std, self.cur_sequence_length,
                                               train=False)
                iv_all_pred, iv_all_loss = self.iv_cvae(iv_batch_sample, z_tta, self.p_std, self.cur_sequence_length,
                                                        train=False)

                pred_list, _, contact_list, bvh_list = all_pred
                # pred_list, pred_contact_list, bvh_list = all_pred
                root_loss, contact_loss, quaternion_loss, position_loss = all_loss

                iv_pred_list, _, iv_contact_list, iv_bvh_list = iv_all_pred
                # iv_pred_list, iv_pred_contact_list, iv_bvh_list = iv_all_pred
                iv_root_loss, iv_contact_loss, iv_quaternion_loss, iv_position_loss = iv_all_loss

                pred_list = blend(pred_list, iv_pred_list)
                contact_list = blend(contact_list, iv_contact_list)

                # preprocess contact
                pred_contact_list = torch.cat(contact_list, 0)
                pred_contact_list = pred_contact_list.view(len(contact_list), -1, 4).permute(1, 0, 2)  # BxTx4
                pred_contact_list[pred_contact_list > 0.5] = 1
                pred_contact_list[pred_contact_list <= 0.5] = 0

                # pred_list = blend(pred_contact_list, iv_pred_contact_list)

                total_loss = (self.train_conf["quaternion_loss_weight"] * (quaternion_loss + iv_quaternion_loss) / 2 +
                              self.train_conf["contact_loss_weight"] * (contact_loss + iv_contact_loss) / 2 +
                              self.train_conf["root_loss_weight"] * (root_loss + iv_root_loss) / 2 +
                              self.train_conf["position_loss_weight"] * (position_loss + iv_position_loss) / 2)

                batch_loss.append(total_loss.cpu().detach())

                # if False:
                #     # save tensor
                #     np.savez("./npz/" + batch_idx.__str__() + "inverse_gt.npz",
                #              batch_sample["global_position"].detach().cpu().numpy())
                #     pred = torch.cat(pred_list, 0)
                #     pred = pred.view(len(pred_list), -1, 22, 3)
                #     np.savez("./npz/" + batch_idx.__str__() + "inverse_pred.npz", pred.detach().cpu().numpy())

                # cal metric
                if self.data_conf["save_metric"]:
                    # ADE & FDE
                    pred = torch.cat(pred_list, 0)
                    pred = pred.view(len(pred_list), -1, 22, 3).permute(1, 0, 2, 3)  # BxTxJx3
                    ADE, FDE = cal_ADE_FDE_metric(pred, batch_sample["global_position"].cuda())
                    F5DE = cal_F5DE_metric(pred, batch_sample["global_position"].cuda())
                    print(batch_idx, " ADE: ", ADE, " FDE: ", FDE, " F5DE: ", F5DE)

                    # NPSS: global position
                    pred_gp = pred.view(pred.size(0), pred.size(1), -1)
                    gt_gp = batch_sample["global_position"].view(batch_sample["global_position"].size(0),
                                                                 batch_sample["global_position"].size(1),
                                                                 -1)
                    NPSS = fast_npss(gt_gp, pred_gp.cpu())
                    print(batch_idx, " NPSS-gp: ", NPSS)

                    # foot sliding
                    pred_pos = torch.cat([x.reshape(x.size(0), -1).unsqueeze(-1) for x in pred_list], -1)
                    pred_vel = (pred_pos[:, self.data_conf["foot_index"], 1:] -
                                pred_pos[:, self.data_conf["foot_index"], :-1])
                    pred_vel = pred_vel.view(pred_vel.size(0), 4, 3, pred_vel.size(-1))
                    # B*4[foot joint num]*3*T[49]  contact:B*T*4(B*4*T[49]*1)
                    dis = torch.abs(pred_vel * pred_contact_list[:, :-1].permute(0, 2, 1).unsqueeze(2))
                    dis = np.asarray(dis.permute(0, 3, 1, 2).cpu())  # B*49*4*3
                    foot_slide = np.linalg.norm(dis, ord=2, axis=3).sum(-1).sum(-1).mean() / self.cur_sequence_length
                    print(batch_idx, " foot-slide : ", foot_slide)
                    # print(pred_contact_list[6, :, 0])

                    metrics.append([ADE, FDE, NPSS, foot_slide, F5DE])

                # save bvh
                if self.data_conf["save_bvh"]:
                    bvh_data = torch.cat([item[6].unsqueeze(0) for item in bvh_list], 0).detach().cpu().numpy()
                    write_to_bvhfile(bvh_data, ('./results/bvh/test_%03d.bvh' % batch_idx),
                                     self.data_conf['joints_to_remove'])

                # save img
                if self.data_conf["save_gif"]:
                    bs = 6
                    img_list = []
                    global_position = batch_sample["global_position"]
                    for t in range(len(pred_list)):
                        pred_global_position = pred_list[t]
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
                    imageio.mimsave(('./results/gif/img_%03d.gif' % batch_idx), img_list, duration=0.1)

                    print("Quaternion Loss = {:.9f}".format(quaternion_loss) +
                          "  Contact Loss = {:.9f}".format(contact_loss) +
                          "  Root Loss = {:.9f}".format(root_loss) +
                          "  Position Loss = {:.9f}".format(position_loss) +
                          "  Total Loss = {:.9f}".format(total_loss))

        if self.data_conf["test_nums"] != "single" and self.data_conf["save_metric"]:
            # plt.hist(np.asarray(metrics)[:, 1], density=True, bins=30, alpha=0.5, histtype='stepfilled',
            #          color='steelblue', edgecolor='none')
            # np.savetxt(os.path.join(self.save_path, self.model_conf["model_name"] + "FDE.csv"), np.asarray(metrics)[:, 1])
            # plt.title(self.model_conf["model_name"] + " FDE distribution")
            # plt.savefig(os.path.join(self.save_path, self.model_conf["model_name"] + "_FDE_dis.png"))

            print("-Mean ADE: ", np.mean(np.asarray(metrics)[:, 0]))
            print("-Mean FDE: ", np.mean(np.asarray(metrics)[:, 1]))
            print("-Mean NPSS: ", np.mean(np.asarray(metrics)[:, 2]))
            print("-Mean Foot Slide: ", np.mean(np.asarray(metrics)[:, 3]))
            print("-Mean F5DE: : ", np.mean(np.asarray(metrics)[:, 4]))

        test_loss = np.mean(batch_loss)
        print("Test Loss = {:.9f} ".format(test_loss))
        return test_loss
