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

import matplotlib
import torch.distributed as dist
import torch.multiprocessing as mp
import albumentations as aug
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import models as models
from data.BFDataset import BNPFDataset, BFDataset, BFNPChAugDataset, BFNPChAugSSLDataset
from train import main as train


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


train_csv_path = "stats/bf_valid_split4_numpy.csv"
valid_csv_path = "stats/bf_train_split4_numpy.csv"
dmso_stats_path = "stats/bf_dmso_stats.csv"

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
            aug.MedianBlur(blur_limit=5),
            aug.GaussNoise(var_limit=(0.1, 1.0), p=0.2),
            aug.CoarseDropout(
                max_holes=32, max_height=32, max_width=32, min_height=16, min_width=16
            ),
        ],
        p=0.25,
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
    "dmso",
]


def app(config):
    exp_folder = os.path.join(config["exp_folder"], config["exp_name"], config["exp_mode"])
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    train_dataset = BFNPChAugSSLDataset(
        root=config["data"]["data_folder"],
        csv_file=config["data"]["train_csv_path"],
        moas=moas,
        geo_transform=geo_transforms,
        colour_transform=colour_transforms,
    )

    # valid_dataset = BFNPChAugSSLDataset(
    #     root=config["data"]["data_folder"],
    #     csv_file=valid_csv_path,
    #     moas=moas,
    #     geo_transform=valid_transforms,
    # )
    # test_dataset = BFNPChAugSSLDataset(
    #     root=config["data"]["data_folder"],
    #     csv_file=valid_csv_path,
    #     moas=moas,
    #     geo_transform=valid_transforms,
    # )

    for train_config in config["train_configs"]:
        model_name = train_config["model"]["args"]["model_name"]

        exp_folder_config = os.path.join(
            exp_folder,
            f'{train_config["model"]["type"]}_{model_name}',
        )

        if not os.path.exists(exp_folder_config):
            os.makedirs(exp_folder_config)
        with open(os.path.join(exp_folder_config, "config_exp.json"), "w") as fp:
            json.dump(train_config, fp)

        valid_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            num_workers=8,
            prefetch_factor=8,
            persistent_workers=True,
        )
        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=config["data"]["batch_size"],
        #     num_workers=8,
        #     prefetch_factor=4,
        #     persistent_workers=True,
        # )
        model = getattr(models, train_config["model"]["type"])(**train_config["model"]["args"])

        model = train.run(
            train_config["train"],
            train_dataset,
            valid_loader,
            model,
            exp_folder_config,
        )
        # test.run(train_config["test"], test_loader, model, exp_folder_config)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="SearchFirst config file path")
    argparser.add_argument("-c", "--conf", help="path to configuration file")
    argparser.add_argument("-d", "--data_dir", help="path to dataset file")
    argparser.add_argument("-r", "--random_seed", help="random_seed", default=42, type=int)
    args = argparser.parse_args()
    config_path = args.conf
    data_path = args.data_dir
    random_seed = args.random_seed
    set_random_seed(random_seed)
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    config["data"]["data_folder"] = data_path

    print(config["exp_name"])
    print(config["data"]["data_folder"])
    app(config)
