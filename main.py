import os
import time
import importlib
import argparse

from model.lafan.LaFan1 import LaFan1
from model.utils.skeleton import Skeleton

parser = argparse.ArgumentParser()
parser.add_argument('mode', default="train", help="'train' / 'test'")
parser.add_argument('model', default="cgan", help="model name")
parser.add_argument('local', default="local", help="where to train")
args = parser.parse_args()

MODULE_LIST = ["cgan",
               "cvae",
               "cvae_adp_dis",
               "cvae_adp_inverse",
               "cvae_bi",
               "cvae_lbi",
               "cvae_bi2",
               "cvae_cd",
               "cvae_cc",
               "cvae_ic",
               "cvae_ic2",
               "cvae_gic",
               "cvae_saic",
               "cvae_ssaic"]

if __name__ == '__main__':
    mode = args.mode
    model_name = args.model

    if model_name in MODULE_LIST:
        # import module
        train_conf = importlib.import_module("model.conf.train_" + model_name + "_conf").train_conf
        test_conf = importlib.import_module("model.conf.test_" + model_name + "_conf").test_conf
        Model = importlib.import_module("model.model_" + model_name).Model

        # config
        if mode == "train":
            opt = train_conf

            save_path = opt["network"]["save_path"]
            if args.local != "local":
                save_path = opt["network"]["cfs_save_path"]

            stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) + "@" + opt["model"]["model_name"]
            log_path = os.path.join(save_path, "log/", stamp)
            model_path = os.path.join(save_path, "checkpoint/", stamp)
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            if not os.path.exists(model_path):
                os.mkdir(model_path)

            opt["network"]["log_path"] = log_path
            opt["network"]["save_path"] = model_path
            opt["network"]["load_path"] = model_path
        else:
            opt = test_conf
        print("Loading Config Finished.")

        # train data
        data_path = opt["data"]["data_path"] if args.local == "local" else opt["data"]["cfs_data_path"]
        train_data = LaFan1(data_path, True, window=opt["data"]["sequence_length"],
                            offset=20, flip=opt["data"]["flip"]) if mode == "train" else None
        test_data = LaFan1(data_path, False, window=opt["data"]["sequence_length"],
                           offset=20, flip=opt["data"]["flip"])
        print("Loading Data Finished.")

        # skeleton
        skeleton_lafan = Skeleton(opt["data"]["offsets"], opt["data"]["parents"]).cuda()
        skeleton_lafan.remove_joints(opt['data']["joints_to_remove"])
        print("Initializing Skeleton Finished.")

        # model
        net = Model(opt, skeleton_lafan, train_data, test_data)
        print("Initializing Model Finished.")

        if mode == "test":
            net.load(opt["network"]["epoch_num"])
            net.test()
        elif mode == "train":
            net.train()
        else:
            print("Select train or test.")

    else:
        print("Input the right model name.")
