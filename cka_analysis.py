from collections import OrderedDict

import albumentations as aug
import torch
from captum.attr import LRP, GuidedBackprop
from captum.attr import visualization as viz

from torchvision.utils import make_grid, save_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.VSDataset import VSNPSSLChAugDataset

from utils.cka import CudaCKA
from models.resnet import ResNet
from torch.utils.data import DataLoader
import torch.nn.functional as F

train_f_csv_path = "stats/train_val_test_splits/fl_subset_train_split1_numpy.csv"
train_bf_csv_path = "stats/train_val_test_splits/bf_subset_train_split1_numpy.csv"
train_shift_path = "stats/shift_bf_subset_train.csv"


valid_f_csv_path = "stats/train_val_test_splits/fl_subset_val_split1_numpy.csv"
valid_bf_csv_path = "stats/train_val_test_splits/bf_subset_val_split1_numpy.csv"
val_shift_path = "stats/shift_bf_subset_val.csv"


dmso_stats_path_f = "stats/dmso_stats.csv"
dmso_stats_path_bf = "stats/bf_dmso_stats.csv"

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
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

model_f = ResNet("resnet50", in_channels=5, num_classes=10)
model_checkpoint = torch.load(
    "/proj/haste_berzelius/exps/specs_phil/fl_exps_1_split5/fl_10cls_chaug/BaseResNet_resnet50/model_best_accuracy.pth"
)
state_dict = OrderedDict()
for k, v in model_checkpoint["model_state_dict"].items():
    name = k[7:]  # remove `module.`
    state_dict[name] = v
model_f.load_state_dict(state_dict)

model_bf = ResNet("resnet50", in_channels=6, num_classes=10)
model_checkpoint = torch.load(
    "/proj/haste_berzelius/exps/specs_phil/bf_exps_1_split5/bf_10cls_chaug_1000e/BaseResNet_resnet50/model_best_accuracy.pth"
)
state_dict = OrderedDict()
for k, v in model_checkpoint["model_state_dict"].items():
    name = k[7:]  # remove `module.`
    state_dict[name] = v
model_bf.load_state_dict(state_dict)

cuda_cka = CudaCKA("cpu")

ckas = []
features_f = []
features_bf = []

with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
        f_in = sample[0]
        bf_in = sample[1]
        target = sample[2]

        op_f, feature_map_f = model_f(f_in)
        op_bf, feature_map_bf = model_bf(bf_in)

        feature_map_f = F.adaptive_avg_pool2d(feature_map_f, (1, 1)).flatten(1).detach()
        feature_map_bf = F.adaptive_avg_pool2d(feature_map_bf, (1, 1)).flatten(1).detach()

        features_f.append(feature_map_f)
        features_bf.append(feature_map_bf)
        print(batch_idx, len(test_loader))

    features_f = torch.cat(features_f)
    features_bf = torch.cat(features_bf)
    linear_cka = cuda_cka.linear_CKA(feature_map_f, feature_map_bf).item()
    kernel_cka = cuda_cka.kernel_CKA(feature_map_f, feature_map_bf, sigma=0.8).item()
    print(f"Linear CKA: {linear_cka:.3f}")
    print(f"Kernel CKA: {kernel_cka:.3f}")
    x = 1
