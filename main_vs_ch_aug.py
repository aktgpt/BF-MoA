import argparse
import json
import os
import random

import numpy as np

# import setproctitle
import torch

os.environ["DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import json
import os
import shutil
from test import main as test

import albumentations as aug
import matplotlib
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import models as models
from data.VSDataset import (
    VSDataset,
    VSNPDataset,
    VSNPMTLDataset,
    VSNPMTLChAugDataset,
    VSNPSSLChAugDataset,
)
from train import main as train


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


train_f_csv_path = "stats/f_numpy_subset_train.csv"
train_bf_csv_path = "stats/bf_numpy_subset_train.csv"
train_shift_path = "stats/shift_bf_subset_train.csv"

val_f_csv_path = "stats/f_numpy_subset_val.csv"
val_bf_csv_path = "stats/bf_numpy_subset_val.csv"
val_shift_path = "stats/shift_bf_subset_val.csv"

dmso_stats_path_f = "stats/dmso_stats.csv"
dmso_stats_path_bf = "stats/bf_dmso_stats.csv"

# train_transforms_f = aug.Compose(
#     [
#         aug.RandomCrop(1024, 1024),
#         aug.Resize(512, 512),
#         aug.Flip(),
#         aug.RandomRotate90(),
#         aug.GaussianBlur(p=0.2),
#         aug.CoarseDropout(max_height=32, max_width=32, p=0.5),
#         aug.RandomGridShuffle(grid=(4, 4), p=0.2),
#     ]
# )
geo_transforms = aug.Compose(
    [
        aug.RandomCrop(1024, 1024),
        aug.Resize(512, 512),
        aug.RandomGridShuffle(grid=(3, 3)),
        aug.Flip(),
        aug.RandomRotate90(),
    ]
)
colour_transforms = aug.PerChannel(
    aug.OneOf(
        [
            aug.GaussianBlur(p=0.2),
            aug.MotionBlur(p=0.2),
            aug.GaussNoise(p=0.2),
            aug.CoarseDropout(max_height=32, max_width=32, p=0.2),
        ],
        p=0.2,
    ),
    p=0.5,
)
valid_transforms = aug.Compose([aug.Resize(1024, 1024)])

moas = [
    "Aurora kinase inhibitor",
    "tubulin polymerization inhibitor",
    "JAK inhibitor",
    "protein synthesis inhibitor",
    "HDAC inhibitor",
    "topoisomerase inhibitor",
    "PARP inhibitor",
    "ATPase inhibitor",
    "retinoid receptor agonist",
    "HSP inhibitor",
]


def app(config):
    exp_folder = os.path.join(config["exp_folder"], config["exp_name"], config["exp_mode"])
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    train_dataset = VSNPSSLChAugDataset(
        root=config["data"]["data_folder"],
        bf_csv_file=train_bf_csv_path,
        f_csv_file=train_f_csv_path,
        shift_csv_file=train_shift_path,
        moas=moas,
        geo_transform=geo_transforms,
        colour_transform=colour_transforms,
    )

    valid_dataset = VSNPSSLChAugDataset(
        root=config["data"]["data_folder"],
        bf_csv_file=val_bf_csv_path,
        f_csv_file=val_f_csv_path,
        shift_csv_file=val_shift_path,
        moas=moas,
        geo_transform=valid_transforms,
    )
    test_dataset = VSNPSSLChAugDataset(
        root=config["data"]["data_folder"],
        bf_csv_file=val_bf_csv_path,
        f_csv_file=val_f_csv_path,
        shift_csv_file=val_shift_path,
        moas=moas,
        geo_transform=valid_transforms,
    )

    for train_config in config["train_configs"]:
        model_name = train_config["model1"]["args"]["model_name"]

        exp_folder_config = os.path.join(
            exp_folder,
            f'{train_config["model1"]["type"]}_{model_name}',
        )

        if not os.path.exists(exp_folder_config):
            os.makedirs(exp_folder_config)
        with open(os.path.join(exp_folder_config, "config_exp.json"), "w") as fp:
            json.dump(train_config, fp)

        # valid_loader = DataLoader(
        #     valid_dataset,
        #     batch_size=config["data"]["batch_size"],
        #     num_workers=8,
        #     prefetch_factor=8,
        #     persistent_workers=True,
        #     pin_memory=True,
        # )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["data"]["batch_size"],
            num_workers=8,
            prefetch_factor=8,
            persistent_workers=True,
        )
        model_list = [
            getattr(models, train_config["model1"]["type"])(**train_config["model1"]["args"]),
            getattr(models, train_config["model2"]["type"])(**train_config["model2"]["args"]),
        ]

        train.run(
            train_config["train"],
            train_dataset,
            valid_dataset,
            model_list,
            exp_folder_config,
        )
        test.run(train_config["test"], test_loader, model_list, exp_folder_config)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="SearchFirst config file path")
    argparser.add_argument("-c", "--conf", help="path to configuration file")
    argparser.add_argument("-d", "--data_dir", help="path to dataset file")
    argparser.add_argument("-r", "--random_seed", help="random_seed", default=42, type=int)

    # args = argparser.parse_args(
    #     ["-c", "configs/vs_dual_attn.json", "-d", "/proj/haste_berzelius/datasets/specs"]
    # )

    args = argparser.parse_args()
    config_path = args.conf
    data_path = args.data_dir
    random_seed = args.random_seed
    set_random_seed(random_seed)
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    config["data"]["data_folder"] = data_path

    print(config["exp_name"] + "_" + config["exp_mode"])
    print(config["data"]["data_folder"])
    app(config)
