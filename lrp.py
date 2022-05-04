import os
from collections import OrderedDict

import albumentations as aug
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import LRP, GuidedBackprop
from captum.attr import visualization as viz
from torchvision.utils import make_grid, save_image

from data.VSDataset import VSNPSSLChAugDataset
from models.resnet import ResNetLRP
from utils.cka import CudaCKA


def dmso_normalize(dmso_stats_df, input, plate):
    dmso_mean = []
    dmso_std = []
    for i in range(1, input.shape[1] + 1):
        dmso_mean.append(dmso_stats_df[plate]["C" + str(i)]["m"])
        dmso_std.append(dmso_stats_df[plate]["C" + str(i)]["std"])
    dmso_mean = torch.tensor(dmso_mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    dmso_std = torch.tensor(dmso_std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    img = (input * dmso_std) + dmso_mean
    img = (img - img.min()) / (img.max() - img.min())
    return img


save_folder = "/proj/haste_berzelius/exps/specs_phil/xai_split2/gbp"

train_f_csv_path = "stats/f_numpy_subset_train_split2.csv"
train_bf_csv_path = "stats/bf_numpy_subset_train_split2.csv"
train_shift_path = "stats/shift_bf_subset_train.csv"


valid_f_csv_path = "stats/f_numpy_subset_valid_split2.csv"
valid_bf_csv_path = "stats/bf_numpy_subset_valid_split2.csv"
val_shift_path = "stats/shift_bf_subset_val.csv"


dmso_stats_path_f = "stats/dmso_stats.csv"
dmso_stats_path_bf = "stats/bf_dmso_stats.csv"

f_dmso_stats = pd.read_csv(dmso_stats_path_f, header=[0, 1], index_col=0)
bf_dmso_stats = pd.read_csv(dmso_stats_path_bf, header=[0, 1], index_col=0)

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

valid_transforms = aug.Compose([aug.Resize(1024, 1024)])

test_dataset = VSNPSSLChAugDataset(
    root="/proj/haste_berzelius/datasets/specs",
    bf_csv_file=valid_bf_csv_path,
    f_csv_file=valid_f_csv_path,
    shift_csv_file=val_shift_path,
    moas=moas,
    geo_transform=valid_transforms,
)

model_f = ResNetLRP("resnet50", in_channels=5, num_classes=10)
model_checkpoint = torch.load(
    "/proj/haste_berzelius/exps/specs_phil/fl_exps_1_split2/fl_10cls_chaug_bfsites_1000e/BaseResNet_resnet50/model_best_accuracy.pth"
)
state_dict = OrderedDict()
for k, v in model_checkpoint["model_state_dict"].items():
    name = k[7:]  # remove `module.`
    state_dict[name] = v
model_f.load_state_dict(state_dict)

model_bf = ResNetLRP("resnet50", in_channels=6, num_classes=10)
model_checkpoint = torch.load(
    "/proj/haste_berzelius/exps/specs_phil/bf_exps_1_split2/bf_10cls_chaug_1000epochs/BaseResNet_resnet50/model_best_accuracy.pth"
)
state_dict = OrderedDict()
for k, v in model_checkpoint["model_state_dict"].items():
    name = k[7:]  # remove `module.`
    state_dict[name] = v
model_bf.load_state_dict(state_dict)

lrp_f = GuidedBackprop(model_f)
lrp_bf = GuidedBackprop(model_bf)

for i in range(len(test_dataset)):
    sample = test_dataset.__getitem__(i)
    f_in = sample[0].unsqueeze(0)
    bf_in = sample[1].unsqueeze(0)
    target = sample[2]
    plate = sample[3]

    attribution_f = lrp_f.attribute(f_in, target=torch.tensor(target).unsqueeze(0))
    attribution_bf = lrp_bf.attribute(bf_in, target=torch.tensor(target).unsqueeze(0))

    img_f = dmso_normalize(f_dmso_stats, f_in, plate)
    img_bf = dmso_normalize(bf_dmso_stats, bf_in, plate)

    viz.visualize_image_attr_multiple(
        np.transpose(attribution_f.squeeze().cpu().detach().numpy(), (1, 2, 0))[:, :, :3],
        np.transpose(img_f.squeeze().cpu().detach().numpy(), (1, 2, 0))[:, :, :3],
        ["original_image", "heat_map"],
        ["all", "all"],
        show_colorbar=True,
        fig_size=(16, 12),
        outlier_perc=2,
    )
    plt.savefig(os.path.join(save_folder, f"f_attr_gbp_{i+1}.png"), dpi=500)

    viz.visualize_image_attr_multiple(
        np.transpose(attribution_bf.squeeze().cpu().detach().numpy(), (1, 2, 0))[:, :, :3],
        np.transpose(img_bf.squeeze().cpu().detach().numpy(), (1, 2, 0))[:, :, :3],
        ["original_image", "heat_map"],
        ["all", "all"],
        show_colorbar=True,
        fig_size=(16, 16),
        outlier_perc=2,
    )
    plt.savefig(os.path.join(save_folder, f"bf_attr_gbp_{i+1}.png"), dpi=500)

# input = make_grid(input, normalize=True).permute(1, 2, 0).cpu().numpy()[:, :, :3]
# attribution1 = make_grid(attribution.detach()).permute(1, 2, 0).cpu().numpy()

# attribution2 = (attribution1 - np.min(attribution1)) / (np.max(attribution1) - np.min(attribution1))
# for i in range(attribution1.shape[-1]):
#     attr_ch = attribution2[:, :, i]
#     cm = plt.get_cmap("jet")
#     attr_ch1 = cm(attr_ch)[:, :, :3]
#     cv2.imwrite(f"attr{i}.png", (attr_ch1 * 255).astype("uint8"))
