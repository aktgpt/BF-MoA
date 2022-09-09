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
from test import main as test


import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import models as models
import data as datasets
from data.bf_dataset import BFDataset
from train import main as train
from scipy import signal
from utils.filecopyfast import ThreadedCopy
from utils.get_aug import get_aug


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def transfer_files(orig_image_path, file_list, dst_dir, bg_gen=False):
    print("Copying files to {}".format(dst_dir))
    img_paths = []
    new_img_paths = []
    for file in file_list:
        df = pd.read_csv(file)
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            if bg_gen:
                img_path = orig_image_path + os.path.splitext(row.path)[0] + "_bg_corrected.npy"
            else:
                img_path = orig_image_path + row.path

            new_image_folder = os.path.split(img_path)[0].replace(orig_image_path, dst_dir)
            if not os.path.exists(new_image_folder):
                os.makedirs(new_image_folder)

            new_img_path = dst_dir + row.path
            img_paths.append(img_path)
            new_img_paths.append(new_img_path)

    ThreadedCopy(img_paths, new_img_paths)
    print("Done copying files")


def app(config):
    exp_folder = os.path.join(config["exp_folder"], config["exp_name"], config["exp_mode"])
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    transfer_files(
        config["data_path"],
        [
            config["data"]["train_csv_path"],
            config["data"]["val_csv_path"],
            config["data"]["test_csv_path"],
        ],
        config["data"]["data_folder"],
        bg_gen=config["data"]["bg_correct"],
    )

    geo_transforms, colour_transforms, valid_transforms = get_aug(config["data"]["aug_type"])

    train_dataset = getattr(datasets, config["data"]["dataset"])(
        root=config["data"]["data_folder"],
        csv_file=config["data"]["train_csv_path"],
        normalize=config["data"]["normalization"],
        dmso_stats_path=config["data"]["dmso_stats_path"],
        moas=config["data"]["moas"],
        geo_transform=geo_transforms,
        colour_transform=colour_transforms,
        bg_correct=config["data"]["bg_correct"],
        modality=config["data"]["modality"],
    )

    valid_dataset = getattr(datasets, config["data"]["dataset"])(
        root=config["data"]["data_folder"],
        csv_file=config["data"]["val_csv_path"],
        normalize=config["data"]["normalization"],
        dmso_stats_path=config["data"]["dmso_stats_path"],
        moas=config["data"]["moas"],
        geo_transform=valid_transforms,
        bg_correct=config["data"]["bg_correct"],
        modality=config["data"]["modality"],
    )

    test_dataset = getattr(datasets, config["data"]["dataset"])(
        root=config["data"]["data_folder"],
        csv_file=config["data"]["test_csv_path"],
        normalize=config["data"]["normalization"],
        dmso_stats_path=config["data"]["dmso_stats_path"],
        moas=config["data"]["moas"],
        geo_transform=valid_transforms,
        bg_correct=config["data"]["bg_correct"],
        modality=config["data"]["modality"],
    )

    model_name = config["model"]["args"]["model_name"]
    exp_folder_config = os.path.join(exp_folder, f'{config["model"]["type"]}_{model_name}')

    if not os.path.exists(exp_folder_config):
        os.makedirs(exp_folder_config)
    with open(os.path.join(exp_folder_config, "config_exp.json"), "w") as fp:
        json.dump(config, fp)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["data"]["batch_size"],
        num_workers=8,
        prefetch_factor=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        num_workers=8,
        prefetch_factor=8,
        persistent_workers=True,
    )
    model = getattr(models, config["model"]["type"])(**config["model"]["args"])

    train.run(config["train"], train_dataset, valid_loader, model, exp_folder_config)
    test.run(config, test_loader, model, exp_folder_config)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="SearchFirst config file path")
    argparser.add_argument("-c", "--conf", help="path to configuration file")
    argparser.add_argument(
        "-d",
        "--data_dir",
        help="path to dataset file",
        default="/proj/haste_berzelius/datasets/specs",
    )
    argparser.add_argument("-r", "--random_seed", help="random_seed", default=42, type=int)
    args = argparser.parse_args(
        # ["-c", "configs/bf_bgcorrect.json", "-d", "/proj/haste_berzelius/datasets/specs"]
    )

    config_path = args.conf
    data_path = args.data_dir
    random_seed = args.random_seed

    set_random_seed(random_seed)

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    config["data"]["data_folder"] = data_path

    print(config["exp_name"])
    print(config["exp_mode"])
    print(config["data"]["data_folder"])
    app(config)
