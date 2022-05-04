import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import utils.colormaps as cmaps
from numpy.lib.shape_base import apply_along_axis
from PIL import Image
from torch import nn, optim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from utils.metrics import get_metrics
from torch.multiprocessing import Lock
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


class ABNTester:
    def __init__(self, config, save_folder):
        self.config = config
        try:
            self.multilabel = config["multilabel"]
        except:
            self.multilabel = False
        try:
            self.neg_attn = config["neg_attn"]
        except:
            self.neg_attn = False
        try:
            self.retrain = config["retrain"]
        except:
            self.retrain = False

        self.save_folder = save_folder
        self.model_checkpoint = torch.load(os.path.join(save_folder, config["model_path"]))
        self.image_folder = os.path.join(
            self.save_folder, "retest_images" if self.retrain else "test_images"
        )

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def test(self, test_dataloader, model):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            self.model = nn.DataParallel(model.cuda())
        else:
            self.model = nn.DataParallel(model.cuda())  # model.cuda()

        self.model.load_state_dict(
            self.model_checkpoint["model_state_dict"],
        )
        best_train_epoch = self.model_checkpoint["epoch"]
        best_train_accuracy = self.model_checkpoint["epoch_accuracy"]
        print(f"Loading model with Epoch:{best_train_epoch},Accuracy:{best_train_accuracy}")

        self._test_epoch(test_dataloader)

    def _test_epoch(self, dataloader):
        self.model.eval()
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")

        outputs = []
        targets = []

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1].cuda().to(non_blocking=True)

                output, output_attn, attn = self.model(input)

                if batch_idx % 5 == 0:
                    save_attn(batch_idx, input, attn, self.image_folder)

                _, preds = torch.max(output, 1)

                outputs.append(preds.detach())
                targets.append(target.detach())

                outer.update(1)

        outputs = torch.cat(outputs).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        f1_macro = f1_score(targets, outputs, average="macro")
        pd.DataFrame([f1_macro]).to_csv(os.path.join(self.save_folder, "f1_macro.csv"))

        f1_micro = f1_score(targets, outputs, average="micro")
        pd.DataFrame([f1_micro]).to_csv(os.path.join(self.save_folder, "f1_micro.csv"))

        confusion_mat = confusion_matrix(targets, outputs)
        pd.DataFrame(confusion_mat).to_csv(os.path.join(self.save_folder, "confusion_matrix.csv"))

        precision = precision_score(targets, outputs, average="macro")
        pd.DataFrame([precision]).to_csv(os.path.join(self.save_folder, "precision.csv"))

        recall = recall_score(targets, outputs, average="macro")
        pd.DataFrame([recall]).to_csv(os.path.join(self.save_folder, "recall.csv"))

        accuracy = balanced_accuracy_score(targets, outputs)
        pd.DataFrame([accuracy]).to_csv(os.path.join(self.save_folder, "avg_accuracy.csv"))

        accuracy = accuracy_score(targets, outputs)
        pd.DataFrame([accuracy]).to_csv(os.path.join(self.save_folder, "acc_test.csv"))


def save_attn(batch_idx, input, attn, save_folder):
    shape = input.shape[2:]
    input = make_grid(input, normalize=True).permute(1, 2, 0).cpu().numpy()[:, :, :3]
    cv2.imwrite(
        os.path.join(save_folder, f"{batch_idx}_input.png"),
        cv2.cvtColor((input * 255).astype("uint8"), cv2.COLOR_BGR2RGB),
    )

    attn = (
        make_grid(F.interpolate(attn, shape, mode="bilinear", align_corners=False))
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    cm = plt.get_cmap("jet")
    attn = cm(attn[:, :, 0])[:, :, :3]
    cv2.imwrite(
        os.path.join(save_folder, f"{batch_idx}_attn.png"),
        cv2.cvtColor((attn * 255).astype("uint8"), cv2.COLOR_BGR2RGB),
    )
    a = (
        0.5
        * (
            cv2.cvtColor(
                cv2.cvtColor(
                    (input * 255).astype("uint8"),
                    cv2.COLOR_RGB2GRAY,
                ),
                cv2.COLOR_GRAY2RGB,
            )
            / 255
        )
        + 0.5 * attn
    )
    cv2.imwrite(
        os.path.join(save_folder, f"{batch_idx}_merged.png"),
        cv2.cvtColor((a * 255).astype("uint8"), cv2.COLOR_BGR2RGB),
    )


def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    b, c, h, w = x.size()
    x = F.relu(x)
    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))

    return F.relu(x - epsilon) / (max_value + epsilon)
