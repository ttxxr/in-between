import os
import logging

import torch.optim as optim
import torch.utils
import torch.utils.cpp_extension
import torch.utils.data as tordata
import matplotlib

from tqdm import tqdm
from model.network.cgan.cgan import *
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")


class Model(object):
    def __init__(self, opt, skeleton, train_data=None, test_data=None):
        self.data_conf = opt["data"]
        self.model_conf = opt["model"]
        self.train_conf = opt["network"]

        self.model_name = self.model_conf["model_name"]
        self.lr = self.train_conf["learning_rate"]
        self.epoch_num = self.train_conf["epoch"]
        self.batch_size = self.train_conf["batch_size"]
        self.save_path = self.train_conf["save_path"]
        self.load_path = self.train_conf["load_path"]

        self.train_data = train_data
        self.test_data = test_data
        self.cur_sequence_length = self.test_data.cur_sequence_length
        self.std = None

        # build cgan
        self.cgan = CGAN(opt, skeleton)

        # build discriminator network
        self.long_discriminator = LongMotionDiscriminator(length=self.data_conf["short_length"],
                                                          in_dim=self.data_conf["joint_num"] * 3 * 2).cuda()
        self.short_discriminator = ShortMotionDiscriminator(length=self.data_conf["short_length"],
                                                            in_dim=self.data_conf["joint_num"] * 3 * 2).cuda()

        # build optimizer
        self.optimizer = optim.Adam(lr=self.lr,
                                    params=(list(self.cgan.parameters())),
                                    betas=(self.train_conf["beta1"], self.train_conf["beta2"]),
                                    weight_decay=self.train_conf["weight_decay"])
        self.optimizer_discriminator = optim.Adam(lr=self.lr * 0.1,
                                                  params=(list(self.long_discriminator.parameters()) +
                                                          list(self.short_discriminator.parameters())),
                                                  betas=(self.train_conf["beta1"], self.train_conf["beta2"]),
                                                  weight_decay=self.train_conf["weight_decay"])

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

    def load(self, epoch=0):
        epoch = "" if epoch == 0 else str(epoch)

        self.cgan.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "cgan.pth")))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.train_conf["load_path"], epoch + "optimizer.pth")))

        # discriminator
        self.short_discriminator.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "short_discriminator.pth")))
        self.long_discriminator.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "long_discriminator.pth")))
        self.optimizer_discriminator.load_state_dict(
            torch.load(os.path.join(self.train_conf["load_path"], epoch + "optimizer_d.pth")))

    def save(self, epoch=""):
        torch.save(self.cgan.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "cgan.pth"))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "optimizer.pth"))

        # discriminator
        torch.save(self.short_discriminator.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "short_discriminator.pth"))
        torch.save(self.long_discriminator.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "long_discriminator.pth"))
        torch.save(self.optimizer_discriminator.state_dict(),
                   os.path.join(self.train_conf["save_path"], str(epoch) + "optimizer_d.pth"))

    def train(self):
        writer = SummaryWriter(self.train_conf["log_path"])
        train_loader = tordata.DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=4,
                                          shuffle=True)
        # save global position std
        self.std = torch.from_numpy(self.train_data.p_std).view(1, 1, -1, 3).cuda()
        np.savetxt(os.path.join(self.save_path, "norm.csv"), self.train_data.p_std[0])

        min_loss = 1000000
        for epoch in range(self.epoch_num):
            self.cgan.train()

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
            for batch_sample in tqdm(train_loader, ncols=100):
                # forward
                all_pred, all_loss = self.cgan(batch_sample, z_tta, self.std, self.cur_sequence_length)

                pred_list = all_pred[0]
                root_loss, contact_loss, quaternion_loss, position_loss = all_loss

                # adversarial input
                fake_position = torch.cat([item.reshape(item.size(0), -1).unsqueeze(-1) for item in pred_list], -1)
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

                discriminator_loss = self.train_conf["gan_loss_weight"] * (short_d_loss + long_d_loss)
                discriminator_loss.backward()
                self.optimizer_discriminator.step()

                # generator
                self.optimizer.zero_grad()
                short_g_fake_logic = torch.mean(self.short_discriminator(fake_input)[:, 0], 1)
                short_g_loss = torch.mean((short_g_fake_logic - 1) ** 2)
                long_g_fake_logic = torch.mean(self.long_discriminator(fake_input)[:, 0], 1)
                long_g_loss = torch.mean((long_g_fake_logic - 1) ** 2)

                # sum loss
                total_loss = (self.train_conf["quaternion_loss_weight"] * quaternion_loss +
                              self.train_conf["contact_loss_weight"] * contact_loss +
                              self.train_conf["root_loss_weight"] * root_loss +
                              self.train_conf["position_loss_weight"] * position_loss +
                              self.train_conf["gan_loss_weight"] * (short_g_loss + long_g_loss))

                total_loss.backward()

                # gradient clip
                nn.utils.clip_grad_norm_(self.cgan.parameters(), 0.5)

                self.optimizer.step()

                # generator loss
                writer.add_scalar("root_loss", root_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("contact_loss", contact_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("position_loss", position_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("quaternion_loss", quaternion_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("total_generator_loss", total_loss, global_step=epoch * 317 + batch_idx)

                # GAN loss
                writer.add_scalar("short_generator_loss", short_g_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("long_generator_loss", long_g_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("short_discriminator_loss", short_d_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("long_discriminator_loss", long_d_loss, global_step=epoch * 317 + batch_idx)
                writer.add_scalar("discriminator_loss", discriminator_loss, global_step=epoch * 317 + batch_idx)

                batch_loss.append(total_loss.cpu().detach())
                batch_idx += 1

            # save model
            cur_train_loss = np.mean(batch_loss)
            if cur_train_loss < min_loss:
                min_loss = cur_train_loss
                self.save()

            # test
            if epoch % 10 == 0:
                test_loss = self.test()
                writer.add_scalar("test loss", test_loss, global_step=epoch * 317)

            # save
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
                                         shuffle=True)

        # load global position std
        self.std = torch.from_numpy(np.loadtxt(os.path.join(self.load_path, "norm.csv"))).view(1, 1, -1, 3).cuda()

        self.cgan.eval()
        self.cur_sequence_length = self.data_conf["sequence_length"]

        # Ztta embedding
        z_tta = gen_ztta(self.cur_sequence_length, self.model_conf["encoder"]["state_dim"][-1],
                         self.model_conf["position_encoding_basis"]).cuda()

        batch_loss = []
        for batch_idx, batch_sample in enumerate(test_loader):
            with torch.no_grad():
                all_pred, all_loss = self.cgan(batch_sample, z_tta, self.std, self.cur_sequence_length, train=False)
                pred_list, bvh_list = all_pred
                root_loss, contact_loss, quaternion_loss, position_loss = all_loss

                total_loss = (self.train_conf["quaternion_loss_weight"] * quaternion_loss +
                              self.train_conf["contact_loss_weight"] * contact_loss +
                              self.train_conf["root_loss_weight"] * root_loss +
                              self.train_conf["position_loss_weight"] * position_loss)

                batch_loss.append(total_loss.cpu().detach())

                # save bvh
                if self.data_conf["save_bvh"]:
                    bvh_data = torch.cat([item[6].unsqueeze(0) for item in bvh_list], 0).detach().cpu().numpy()
                    write_to_bvhfile(bvh_data, ('./results/bvh/test_%03d.bvh' % batch_idx),
                                     self.data_conf['joints_to_remove'])

                # save gif
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
                                                  global_position[bs, k].view(22, 3).detach().cpu().numpy(),
                                                  global_position[bs, -1].view(22, 3).detach().cpu().numpy()], 0),
                                  t, './results/img/gt')
                        pred_img = Image.open('./results/img/pred_' + str(t) + '.png', 'r')
                        gt_img = Image.open('./results/img/gt_' + str(t) + '.png', 'r')
                        img_list.append(np.concatenate([pred_img, gt_img.resize(pred_img.size)], 1))

                    import imageio
                    imageio.mimsave(('./results/gif/img_%03d.gif' % batch_idx), img_list, duration=0.1)

        test_loss = np.mean(batch_loss)
        # print("Test Loss = {:.9f} ".format(test_loss))
        return test_loss
