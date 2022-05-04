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


class DualAttnTester:
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
        self.model_checkpoint_f = torch.load(os.path.join(save_folder, config["model1_path"]))
        self.model_checkpoint_bf = torch.load(os.path.join(save_folder, config["model2_path"]))

        self.image_folder_f = os.path.join(self.save_folder, "test_images_f")
        if not os.path.exists(self.image_folder_f):
            os.makedirs(self.image_folder_f)
        self.image_folder_bf = os.path.join(self.save_folder, "test_images_bf")
        if not os.path.exists(self.image_folder_bf):
            os.makedirs(self.image_folder_bf)

    def test(self, test_dataloader, models):

        self.model_f = nn.DataParallel(models[0].cuda())
        self.model_bf = nn.DataParallel(models[1].cuda())

        self.model_f.load_state_dict(self.model_checkpoint_f["model_state_dict"])
        self.model_bf.load_state_dict(self.model_checkpoint_bf["model_state_dict"])

        best_train_epoch = self.model_checkpoint_f["epoch"]
        best_train_accuracy = self.model_checkpoint_f["epoch_accuracy"]
        print(f"Loading FL model with Epoch:{best_train_epoch},Accuracy:{best_train_accuracy}")

        best_train_epoch = self.model_checkpoint_bf["epoch"]
        best_train_accuracy = self.model_checkpoint_bf["epoch_accuracy"]
        print(f"Loading BF model with Epoch:{best_train_epoch},Accuracy:{best_train_accuracy}")

        self._test_epoch(test_dataloader)

        # accuracy = self._test_epoch(test_dataloader)

        # if self.multilabel:
        #     df_acc = pd.DataFrame({"f1": [accuracy]})
        #     df_acc.to_csv(
        #         os.path.join(self.save_folder, "re_f1_test.csv" if self.retrain else "f1_test.csv")
        #     )
        # else:
        #     df_acc = pd.DataFrame({"accuracies": [accuracy]})
        #     df_acc.to_csv(
        #         os.path.join(
        #             self.save_folder, "re_acc_test.csv" if self.retrain else "acc_test.csv"
        #         )
        #     )

    def _test_epoch(self, dataloader):
        self.model_f.eval()
        self.model_bf.eval()

        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")

        outputs_f = []
        outputs_bf = []
        targets = []

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                f_in = sample[0].cuda().to(non_blocking=True)
                bf_in = sample[1].cuda().to(non_blocking=True)
                target = sample[2].cuda().to(non_blocking=True)

                op_f, cam_op_f, attn_f = self.model_f(f_in)
                op_bf, cam_op_bf, attn_bf = self.model_bf(bf_in)

                if batch_idx % 5 == 0:
                    save_attn(batch_idx, f_in, attn_f, self.image_folder_f)
                    save_attn(batch_idx, bf_in, attn_bf, self.image_folder_bf)

                _, preds_f = torch.max(op_f, 1)
                _, preds_bf = torch.max(op_bf, 1)

                outputs_f.append(preds_f.detach())
                outputs_bf.append(preds_bf.detach())
                targets.append(target.detach())

                outer.update(1)

        outputs_f = torch.cat(outputs_f).cpu().numpy()
        outputs_bf = torch.cat(outputs_bf).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        f1_macro_f = f1_score(targets, outputs_f, average="macro")
        pd.DataFrame([f1_macro_f]).to_csv(os.path.join(self.save_folder, "f1_macro_f.csv"))
        f1_macro_bf = f1_score(targets, outputs_bf, average="macro")
        pd.DataFrame([f1_macro_bf]).to_csv(os.path.join(self.save_folder, "f1_macro_bf.csv"))

        f1_micro_f = f1_score(targets, outputs_f, average="micro")
        pd.DataFrame([f1_micro_f]).to_csv(os.path.join(self.save_folder, "f1_micro_f.csv"))
        f1_micro_bf = f1_score(targets, outputs_bf, average="micro")
        pd.DataFrame([f1_micro_bf]).to_csv(os.path.join(self.save_folder, "f1_micro_bf.csv"))

        confusion_mat_f = confusion_matrix(targets, outputs_f)
        pd.DataFrame(confusion_mat_f).to_csv(
            os.path.join(self.save_folder, "confusion_matrix_f.csv")
        )
        confusion_mat_bf = confusion_matrix(targets, outputs_bf)
        pd.DataFrame(confusion_mat_bf).to_csv(
            os.path.join(self.save_folder, "confusion_matrix_bf.csv")
        )

        precision_f = precision_score(targets, outputs_f, average="macro")
        pd.DataFrame([precision_f]).to_csv(os.path.join(self.save_folder, "precision_f.csv"))
        precision_bf = precision_score(targets, outputs_bf, average="macro")
        pd.DataFrame([precision_bf]).to_csv(os.path.join(self.save_folder, "precision_bf.csv"))

        recall_f = recall_score(targets, outputs_f, average="macro")
        pd.DataFrame([recall_f]).to_csv(os.path.join(self.save_folder, "recall_f.csv"))
        recall_bf = recall_score(targets, outputs_bf, average="macro")
        pd.DataFrame([recall_bf]).to_csv(os.path.join(self.save_folder, "recall_bf.csv"))

        avg_accuracy_f = balanced_accuracy_score(targets, outputs_f)
        pd.DataFrame([avg_accuracy_f]).to_csv(os.path.join(self.save_folder, "avg_accuracy_f.csv"))
        avg_accuracy_bf = balanced_accuracy_score(targets, outputs_bf)
        pd.DataFrame([avg_accuracy_bf]).to_csv(
            os.path.join(self.save_folder, "avg_accuracy_bf.csv")
        )

        accuracy_f = accuracy_score(targets, outputs_f)
        pd.DataFrame([accuracy_f]).to_csv(os.path.join(self.save_folder, "accuracy_f.csv"))
        accuracy_bf = accuracy_score(targets, outputs_bf)
        pd.DataFrame([accuracy_bf]).to_csv(os.path.join(self.save_folder, "accuracy_bf.csv"))


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
