import glob
import json
import os
from collections import OrderedDict

import albumentations as aug
import colorcet as cc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
import umap
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import models as models
from data.dataset import MOADataset
from utils.cka.cka_dataloader import CKA


def make_nmf(input, feature_map, batch_idx, nmf_folder):
    _, _, w_im, h_im = input.shape
    images = (
        make_grid(input, normalize=True, nrow=8, padding=0).permute(1, 2, 0).detach().cpu().numpy()
    )[:, :, :3]
    b, c, w, h = feature_map.shape
    n_rows = b // 8
    nmf_images = [[] for i in range(n_rows)]
    for i in range(b):
        a = feature_map.cpu().numpy()[i].reshape(c, w * h).transpose(1, 0)
        a1 = NMF(n_components=3, random_state=0, max_iter=500).fit_transform(a).reshape(w, h, 3)
        a1[:, :, 0] = (a1[:, :, 0] - np.min(a1[:, :, 0])) / (
            np.max(a1[:, :, 0]) - np.min(a1[:, :, 0])
        )
        a1[:, :, 1] = (a1[:, :, 1] - np.min(a1[:, :, 1])) / (
            np.max(a1[:, :, 1]) - np.min(a1[:, :, 1])
        )
        a1[:, :, 2] = (a1[:, :, 2] - np.min(a1[:, :, 2])) / (
            np.max(a1[:, :, 2]) - np.min(a1[:, :, 2])
        )

        if i % 8 == 0:
            nmf_images[i // 8] = cv2.resize(a1, (w_im, h_im))
        else:
            nmf_images[i // 8] = np.hstack([nmf_images[i // 8], cv2.resize(a1, (w_im, h_im))])
    nmf_images = np.concatenate(nmf_images)
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(images)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_aspect("equal")
    ax[1].imshow(nmf_images)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_aspect("equal")
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.savefig(
        os.path.join(nmf_folder, f"{batch_idx}_nmf_images.png"),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def get_feature_df(config, dataloader, model, nmf_folder):
    model.eval()
    outer = tqdm.tqdm(total=len(dataloader), desc="Batches Processed:", position=0)

    outputs = []
    targets = []
    plates = []
    sites = []
    compounds = []
    wells = []

    feature_data = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            target = sample[1].cuda().to(non_blocking=True)
            plate = sample[2]
            site = sample[3]
            compound = sample[4]
            well = sample[5]

            output, feature_map = model(input)

            if batch_idx % 10 == 0:
                make_nmf(input, feature_map, batch_idx, nmf_folder)

            feature_map = torch.flatten(
                torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)), start_dim=1
            )

            _, preds = torch.max(output, 1)

            feature_data.append(feature_map)

            outputs.append(preds.detach())
            targets.append(target.detach())

            plates.extend(plate)
            sites.extend(site)
            compounds.extend(compound)
            wells.extend(well)

            outer.update(1)

    outputs = torch.cat(outputs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    feature_data = torch.cat(feature_data).cpu().numpy()

    moas = np.sort(config["data"]["moas"])
    sample_moas = []
    pred_moas = []
    for i, target in enumerate(targets):
        sample_moas.append(moas[target])
        pred_moas.append(moas[outputs[i]])

    df_dict = {
        "plate": plates,
        "well": wells,
        "site": sites,
        "compound": compounds,
        "moa": sample_moas,
        "pred_moa": pred_moas,
    }
    for i in range(feature_data.shape[1]):
        df_dict[f"fv_{i}"] = feature_data[:, i]

    df = pd.DataFrame(df_dict)
    return df


valid_transforms = aug.Compose([aug.Resize(1080, 1080, p=1.0), aug.CenterCrop(1024, 1024, p=1.0)])


folders = sorted(glob.glob("/proj/haste_berzelius/exps/specs_non_grit_based/*bf*"))

for folder in folders:
    exp_folders = sorted(glob.glob(os.path.join(folder, "*/ResNet_resnet50/")))
    for exp_folder in exp_folders:
        print(exp_folder)

        # if not os.path.isfile(os.path.join(exp_folder, "feature_data_train.csv")):
        config = json.load(open(os.path.join(exp_folder, "config_exp.json")))
        config["data"]["data_folder"] = "/proj/haste_berzelius/datasets/specs"
        if not "mean_mode" in config["data"]:
            config["data"]["mean_mode"] = "mean"
        if not "modality" in config["data"]:
            config["data"]["modality"] = "bf"

        model = getattr(models, config["model"]["type"])(**config["model"]["args"])
        model = model.cuda()  # nn.DataParallel(model.cuda())

        model_checkpoint = torch.load(os.path.join(exp_folder, config["test"]["model_path"]))
        new_state_dict = OrderedDict()
        for k, v in model_checkpoint["model_state_dict"].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        best_train_epoch = model_checkpoint["epoch"]
        best_train_accuracy = model_checkpoint["epoch_accuracy"]
        print(f"Loading model with Epoch:{best_train_epoch},Accuracy:{best_train_accuracy}")

        # train_dataset = MOADataset(
        #     root=config["data"]["data_folder"],
        #     csv_file=config["data"]["train_csv_path"],
        #     normalize=config["data"]["normalization"],
        #     dmso_stats_path=config["data"]["dmso_stats_path"],
        #     moas=config["data"]["moas"],
        #     geo_transform=valid_transforms,
        #     bg_correct=config["data"]["bg_correct"],
        #     modality=config["data"]["modality"],
        #     mean_mode=config["data"]["mean_mode"],
        # )
        # dataloader = DataLoader(
        #     train_dataset,
        #     batch_size=config["data"]["batch_size"],
        #     num_workers=8,
        #     prefetch_factor=4,
        # )
        # nmf_folder = os.path.join(exp_folder, "train_nmf_images")
        # if not os.path.exists(nmf_folder):
        #     os.makedirs(nmf_folder)
        # df = get_feature_df(config, dataloader, model, nmf_folder)
        # df.to_csv(os.path.join(exp_folder, "feature_data_train.csv"), index=False)

        # if not os.path.isfile(os.path.join(exp_folder, "feature_data_test.csv")):
        test_dataset = MOADataset(
            root=config["data"]["data_folder"],
            csv_file=config["data"]["test_csv_path"],
            normalize=config["data"]["normalization"],
            dmso_stats_path=config["data"]["dmso_stats_path"],
            moas=config["data"]["moas"],
            geo_transform=valid_transforms,
            bg_correct=config["data"]["bg_correct"],
            modality=config["data"]["modality"],
            mean_mode=config["data"]["mean_mode"],
        )
        dataloader = DataLoader(
            test_dataset,
            batch_size=config["data"]["batch_size"],
            num_workers=8,
            prefetch_factor=4,
        )
        nmf_folder = os.path.join(exp_folder, "nmf_images")
        if not os.path.exists(nmf_folder):
            os.makedirs(nmf_folder)
        df = get_feature_df(config, dataloader, model, nmf_folder)
        df.to_csv(os.path.join(exp_folder, "feature_data_test.csv"), index=False)
