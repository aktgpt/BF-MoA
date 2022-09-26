import json
import os

import albumentations as aug
import numpy as np
import pandas as pd
import torch
import tqdm
import umap
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE, trustworthiness
import models as models
from data.dataset import MOADataset
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt

config_path = "configs/bf_non_grit.json"

with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())
config["exp_name"] = config["exp_name"].replace("split1", "split2")
config["exp_mode"] = config["exp_mode"].replace("well", "dmso")

config["data"]["normalization"] = "dmso"
config["data"]["test_csv_path"] = config["data"]["test_csv_path"].replace("split1", "split2")

config["data"]["data_folder"] = "/proj/haste_berzelius/datasets/specs"

valid_transforms = aug.Compose([aug.Resize(1080, 1080, p=1.0), aug.CenterCrop(1024, 1024, p=1.0)])
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
    prefetch_factor=8,
    persistent_workers=True,
)

exp_folder = os.path.join(config["exp_folder"], config["exp_name"], config["exp_mode"])
model_name = config["model"]["args"]["model_name"]
exp_folder_config = os.path.join(exp_folder, f'{config["model"]["type"]}_{model_name}')

model = getattr(models, config["model"]["type"])(**config["model"]["args"])
model = nn.DataParallel(model.cuda())
model_checkpoint = torch.load(os.path.join(exp_folder_config, config["test"]["model_path"]))
model.load_state_dict(model_checkpoint["model_state_dict"])

best_train_epoch = model_checkpoint["epoch"]
best_train_accuracy = model_checkpoint["epoch_accuracy"]
print(f"Loading model with Epoch:{best_train_epoch},Accuracy:{best_train_accuracy}")

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

        # if batch_idx % 10 == 0:
        #     self.make_nmf(input, feature_map, batch_idx)

        _, preds = torch.max(output, 1)

        # feat_corr_mat.append(get_feat_corr(feature_map))
        # spatial_corr_mat.append(get_spatial_corr(feature_map))
        # feature_data.append(F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1).detach())

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

df = pd.DataFrame(
    {
        "plate": plates,
        "well": wells,
        "site": sites,
        "compound": compounds,
        "moa": sample_moas,
        "pred_moa": pred_moas,
        "fv": list(feature_data),
    }
)
df = pd.DataFrame(
    {
        "plate": plates,
        "well": wells,
        "site": sites,
        "compound": compounds,
        "moa": sample_moas,
        "pred_moa": pred_moas,
    }
)


umap_embed = umap.UMAP(n_components=3).fit(feature_data)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


fig = plt.figure(figsize=(12, 12))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# get colormap from seaborn
cmap = ListedColormap(sns.color_palette(cc.glasbey, 11).as_hex())
sc = ax.scatter(
    umap_embed.embedding_[:, 0],
    umap_embed.embedding_[:, 1],
    umap_embed.embedding_[:, 2],
    s=20,
    c=targets,
    marker="o",
    cmap=cmap,
    alpha=1,
)
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
plt.legend(*sc.legend_elements())


df["x"] = umap_embed.embedding_[:, 0]
df["y"] = umap_embed.embedding_[:, 1]
trust_umap = trustworthiness(feature_data, umap_embed.embedding_, metric="cosine")


sns.scatterplot(
    x="x",
    y="y",
    hue="moa",
    palette=sns.color_palette(cc.glasbey, len(moas)),
    legend="brief",
    s=100,
    alpha=0.9,
    data=df,
)

tsne_embed = TSNE(n_components=2).fit_transform(feature_data)
df["x"] = tsne_embed[:, 0]
df["y"] = tsne_embed[:, 1]
plt.figure()
sns.scatterplot(
    x="x",
    y="y",
    hue="moa",
    palette=sns.color_palette(cc.glasbey, len(moas)),
    legend="brief",
    s=100,
    alpha=0.9,
    data=df,
)
plt.show()

trust_tsne = trustworthiness(feature_data, tsne_embed, metric="cosine")
# model_perf = pd.read_csv(
#     "/proj/haste_berzelius/exps/specs_non_grit_based/bf_exps_1_split5/bf_11cls_basic_aug_dmsonorm_750e_sgd_bgcorrect/ResNet_resnet50/analysis.csv",
#     index_col=0,
# )

# # a = model_perf[(model_perf.moa =="dmso") | (model_perf.pred_moa == "dmso")]
# moa2label = {moa: i for i, moa in enumerate(model_perf.moa.unique())}

# model_perf["label"] = model_perf.moa.apply(lambda x: moa2label[x])
# model_perf["pred_label"] = model_perf.pred_moa.apply(lambda x: moa2label[x])
# print(
#     classification_report(
#         model_perf.label, model_perf.pred_label, target_names=model_perf.moa.unique()
#     )
# )
x = 1
