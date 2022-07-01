import os
from collections import OrderedDict

import albumentations as aug
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
from captum.attr import LRP, GuidedBackprop
from captum.attr import visualization as viz
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from data.VSDataset import VSNPSSLChAugDataset
from models.resnet import BaseResNet

train_f_csv_path = "stats/f_numpy_subset_train_split5.csv"
train_bf_csv_path = "stats/bf_numpy_subset_train_split5.csv"
train_shift_path = "stats/shift_bf_subset_train.csv"


valid_f_csv_path = "stats/f_numpy_subset_valid_split5.csv"
valid_bf_csv_path = "stats/bf_numpy_subset_valid_split5.csv"
val_shift_path = "stats/shift_bf_subset_val.csv"


dmso_stats_path_f = "stats/dmso_stats.csv"
dmso_stats_path_bf = "stats/bf_dmso_stats.csv"

save_folder_f = "/proj/haste_berzelius/exps/specs_phil/fl_exps_1_split5/fl_10cls_chaug_bfsites_1000e/BaseResNet_resnet50/"
save_folder_bf = "/proj/haste_berzelius/exps/specs_phil/bf_exps_1_split5/bf_10cls_chaug_1000e/BaseResNet_resnet50/"

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

site_conversion = pd.DataFrame(
    {"bf_sites": ["s1", "s2", "s3", "s4", "s5"], "f_sites": ["s2", "s4", "s5", "s6", "s8"]}
)


test_dataset = VSNPSSLChAugDataset(
    root="/proj/haste_berzelius/datasets/specs",
    bf_csv_file=valid_bf_csv_path,
    f_csv_file=valid_f_csv_path,
    shift_csv_file=val_shift_path,
    moas=moas,
    geo_transform=aug.Compose([aug.Resize(1024, 1024)]),
)
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=8, prefetch_factor=2)

model_f = BaseResNet("resnet50", in_channels=5, num_classes=10)
model_checkpoint = torch.load(
    "/proj/haste_berzelius/exps/specs_phil/fl_exps_1_split5/fl_10cls_chaug_bfsites_1000e/BaseResNet_resnet50/model_best_accuracy.pth"
)
state_dict = OrderedDict()
for k, v in model_checkpoint["model_state_dict"].items():
    name = k[7:]  # remove `module.`
    state_dict[name] = v
model_f.load_state_dict(state_dict)

model_bf = BaseResNet("resnet50", in_channels=6, num_classes=10)
model_checkpoint = torch.load(
    "/proj/haste_berzelius/exps/specs_phil/bf_exps_1_split5/bf_10cls_chaug_1000e/BaseResNet_resnet50/model_best_accuracy.pth"
)
state_dict = OrderedDict()
for k, v in model_checkpoint["model_state_dict"].items():
    name = k[7:]  # remove `module.`
    state_dict[name] = v
model_bf.load_state_dict(state_dict)

targets = []
plates = []
bf_sites = []
compounds = []
wells = []
features_f = []
features_bf = []

with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
        f_in = sample[0]
        bf_in = sample[1]
        target = sample[2]
        plate = sample[3]
        bf_site = sample[4]
        compound = sample[5]
        well = sample[6]

        targets.append(target)
        plates.extend(plate)
        bf_sites.extend(bf_site)
        compounds.extend(compound)
        wells.extend(well)

        op_f, feature_map_f = model_f(f_in)
        op_bf, feature_map_bf = model_bf(bf_in)

        feature_map_f = F.adaptive_avg_pool2d(feature_map_f, (1, 1)).flatten(1).detach()
        feature_map_bf = F.adaptive_avg_pool2d(feature_map_bf, (1, 1)).flatten(1).detach()

        features_f.append(feature_map_f)
        features_bf.append(feature_map_bf)
        print(batch_idx, len(test_loader))

    features_f = torch.cat(features_f)
    features_bf = torch.cat(features_bf)
    targets = torch.cat(targets).cpu().numpy()

    sample_moas = []
    for target in targets:
        sample_moas.append(moas[target])
    f_sites = []
    for bf_site in bf_sites:
        f_sites.append(
            site_conversion["f_sites"][np.where(site_conversion["bf_sites"] == bf_site)[0][0]]
        )
    df = pd.DataFrame(
        {
            "plate": plates,
            "well": wells,
            "f_site": f_sites,
            "bf_site": bf_sites,
            "compound": compounds,
            "moa": sample_moas,
            "fv_f": list(features_f.cpu().numpy()),
            "fv_bf": list(features_bf.cpu().numpy()),
        }
    )
    df.to_pickle(os.path.join(save_folder_f, "analysis"))
    df.to_pickle(os.path.join(save_folder_bf, "analysis"))

# trans_f = umap.UMAP(random_state=42, n_components=2).fit(features_f)
# trans_bf = umap.UMAP(random_state=42, n_components=2).fit(features_bf)


# df["x"] = trans_f.embedding_[:, 0]
# df["y"] = trans_f.embedding_[:, 1]

# fig, ax = plt.subplots(1, figsize=(16, 12))
# scatter = sns.scatterplot(
#     x="x", y="y", hue="moa", style="well", palette="deep", legend="brief", data=df
# )
# plt.savefig(os.path.join(save_folder_f, "test_embed_moa_well.png"), dpi=500)
# plt.close()
df = pd.read_pickle(os.path.join(save_folder_f, "analysis"))

fv_f = np.array(df["fv_f"])
features_f = []
for fv in fv_f:
    features_f.append(fv[np.newaxis, ...])
features_f = np.concatenate(features_f)
trans_f = umap.UMAP(random_state=42, n_components=2).fit(features_f)

df["x"] = trans_f.embedding_[:, 0]
df["y"] = trans_f.embedding_[:, 1]

fig, ax = plt.subplots(1, figsize=(16, 12))
scatter = sns.scatterplot(
    x="x", y="y", hue="moa", style="well", palette="deep", legend="brief", data=df
)
plt.savefig(os.path.join(save_folder_f, "test_embed_moa_well.png"), dpi=500)
plt.close()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

df_moa = df[df["moa"] == moas[0]]
fig, ax = plt.subplots(1, figsize=(16, 12))
scatter = sns.scatterplot(
    x="x",
    y="y",
    hue="compound",
    style="well",
    palette="deep",
    legend="full",
    s=100,
    alpha=0.8,
    data=df_moa,
)
plt.xlim(list(xlim))
plt.ylim(list(ylim))
plt.savefig(os.path.join(save_folder_f, f"test_embed_{moas[0]}_well.png"), dpi=500)
plt.close()
