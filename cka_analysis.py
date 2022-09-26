import gc
import glob
import json
import os
from collections import OrderedDict

import albumentations as aug
import pandas as pd
import torch
from torch.utils.data import DataLoader

import models as models
from data.dataset import MOADataset
from utils.cka.cka_dataloader import CKA


def load_model(exp_folder, config):
    model = getattr(models, config["model"]["type"])(**config["model"]["args"])
    # model = model.cuda()  # nn.DataParallel(model.cuda())

    model_checkpoint = torch.load(os.path.join(exp_folder, config["test"]["model_path"]))
    new_state_dict = OrderedDict()
    for k, v in model_checkpoint["model_state_dict"].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    best_train_epoch = model_checkpoint["epoch"]
    best_train_accuracy = model_checkpoint["epoch_accuracy"]
    print(f"Loading model with Epoch:{best_train_epoch},Accuracy:{best_train_accuracy}")
    return model


valid_transforms = aug.Compose([aug.Resize(1080, 1080, p=1.0), aug.CenterCrop(1024, 1024, p=1.0)])


def get_dataloaders(config_fl, config_bf):
    bf_df = (
        pd.read_csv(config_bf["data"]["test_csv_path"])
        .sort_values(["plate", "well", "site"])
        .reset_index(drop=True)
    )
    fl_df = (
        pd.read_csv(config_fl["data"]["test_csv_path"])
        .sort_values(["plate", "well", "bf_site"])
        .reset_index(drop=True)
    )
    for (_, bf_row), (_, fl_row) in zip(bf_df.iterrows(), fl_df.iterrows()):
        assert bf_row["plate"] == fl_row["plate"], "plates do not match"
        assert bf_row["well"] == fl_row["well"], "wells do not match"
        assert bf_row["site"] == fl_row["bf_site"], "sites do not match"

    fl_dataset = MOADataset(
        root=config_fl["data"]["data_folder"],
        csv_file=fl_df,
        normalize=config_fl["data"]["normalization"],
        dmso_stats_path=config_fl["data"]["dmso_stats_path"],
        moas=config_fl["data"]["moas"],
        geo_transform=valid_transforms,
        bg_correct=config_fl["data"]["bg_correct"],
        modality=config_fl["data"]["modality"],
        mean_mode=config_fl["data"]["mean_mode"],
    )

    bf_dataset = MOADataset(
        root=config_bf["data"]["data_folder"],
        csv_file=bf_df,
        normalize=config_bf["data"]["normalization"],
        dmso_stats_path=config_bf["data"]["dmso_stats_path"],
        moas=config_bf["data"]["moas"],
        geo_transform=valid_transforms,
        bg_correct=config_bf["data"]["bg_correct"],
        modality=config_bf["data"]["modality"],
        mean_mode=config_bf["data"]["mean_mode"],
    )
    fl_dataloader = DataLoader(
        fl_dataset, batch_size=16, num_workers=8, prefetch_factor=4, shuffle=False
    )
    bf_dataloader = DataLoader(
        bf_dataset, batch_size=16, num_workers=8, prefetch_factor=4, shuffle=False
    )
    return fl_dataloader, bf_dataloader


bf_folders = sorted(glob.glob("/proj/haste_berzelius/exps/specs_non_grit_based/*bf*"))
fl_folders = sorted(glob.glob("/proj/haste_berzelius/exps/specs_non_grit_based/*fl*"))

for bf_folder, fl_folder in zip(bf_folders, fl_folders):
    bf_exp_folders = sorted(glob.glob(os.path.join(bf_folder, "*/ResNet_resnet50/")))
    bf_exp_folders = [x for x in bf_exp_folders if "11cls" in x]
    bf_exp_folders = [x for x in bf_exp_folders if "basic_aug" in x]

    fl_exp_folder = os.path.join(
        fl_folder, "fl_11cls_basic_aug_dmso_norm_750e_sgd/ResNet_resnet50/"
    )
    fl_exp_folder = os.path.join(
        fl_folder, "fl_11cls_basic_aug_dmso_norm_750e_sgd/ResNet_resnet50/"
    )
    fl_config = json.load(open(os.path.join(fl_exp_folder, "config_exp.json")))
    fl_config["data"]["data_folder"] = "/proj/haste_berzelius/datasets/specs"
    fl_config["data"]["mean_mode"] = "mean"
    fl_config["data"]["modality"] = "fl"

    for bf_exp_folder in bf_exp_folders:
        # if not os.path.isfile(os.path.join(bf_exp_folder, "cka.png")):
        fl_model = load_model(fl_exp_folder, fl_config)

        bf_config = json.load(open(os.path.join(bf_exp_folder, "config_exp.json")))
        bf_config["data"]["data_folder"] = "/proj/haste_berzelius/datasets/specs"
        bf_config["data"]["mean_mode"] = "mean"
        bf_config["data"]["modality"] = "bf"
        bf_model = load_model(bf_exp_folder, bf_config)

        fl_dataloader, bf_dataloader = get_dataloaders(fl_config, bf_config)

        cka = CKA(
            fl_model,
            bf_model,
            model1_name="FL_model",  # good idea to provide names to avoid confusion
            model2_name="BF_model",
            model1_layers=[
                "layer1.0",
                "layer1.1",
                "layer1.2",
                "layer2.0",
                "layer2.1",
                "layer2.2",
                "layer2.3",
                "layer3.0",
                "layer3.1",
                "layer3.2",
                "layer3.3",
                "layer3.4",
                "layer3.5",
                "layer4.0",
                "layer4.1",
                "layer4.2",
            ],
            model2_layers=[
                "layer1.0",
                "layer1.1",
                "layer1.2",
                "layer2.0",
                "layer2.1",
                "layer2.2",
                "layer2.3",
                "layer3.0",
                "layer3.1",
                "layer3.2",
                "layer3.3",
                "layer3.4",
                "layer3.5",
                "layer4.0",
                "layer4.1",
                "layer4.2",
            ],
            device="cuda",
        )
        cka.compare(fl_dataloader, bf_dataloader)
        results = cka.export()
        cka.plot_results(save_path=os.path.join(bf_exp_folder, "cka.png"))
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    del obj
            except: pass
        torch.cuda.empty_cache()
            # if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #     del obj


        # del cka
        # del fl_model
        # del bf_model
        # del fl_dataloader
        # del bf_dataloader
        
    # results_df = pd.DataFrame(results)


# for bf_folder in bf_folders:
#     bf_exp_folders = sorted(glob.glob(os.path.join(bf_folder, "*/ResNet_resnet50/")))
#     for bf_exp_folder in bf_exp_folders:
#         print(bf_exp_folder)

#         # if not os.path.isfile(os.path.join(bf_exp_folder, "feature_data_train.csv")):
#         config = json.load(open(os.path.join(bf_exp_folder, "config_exp.json")))
#         config["data"]["data_folder"] = "/proj/haste_berzelius/datasets/specs"
#         config["data"]["mean_mode"] = "mean"
#         config["data"]["modality"] = "bf"

#         model = load_model(bf_exp_folder, config)

#         dataloader = get_dataloaders(valid_transforms, config)
#         cka = CKA(
#             model,
#             model,
#             model1_name="ResNet18",  # good idea to provide names to avoid confusion
#             model2_name="ResNet34",
#             device="cpu",
#         )
#         cka.compare(dataloader)
#         results = cka.export()
#         df = get_feature_df(config, dataloader, model)
#         df.to_csv(os.path.join(bf_exp_folder, "feature_data_train.csv"), index=False)

#         if not os.path.isfile(os.path.join(bf_exp_folder, "feature_data_test.csv")):
#             test_dataset = MOADataset(
#                 root=config["data"]["data_folder"],
#                 csv_file=config["data"]["test_csv_path"],
#                 normalize=config["data"]["normalization"],
#                 dmso_stats_path=config["data"]["dmso_stats_path"],
#                 moas=config["data"]["moas"],
#                 geo_transform=valid_transforms,
#                 bg_correct=config["data"]["bg_correct"],
#                 modality=config["data"]["modality"],
#                 mean_mode=config["data"]["mean_mode"],
#             )
#             dataloader = DataLoader(
#                 test_dataset,
#                 batch_size=config["data"]["batch_size"],
#                 num_workers=8,
#                 prefetch_factor=4,
#             )
#             df = get_feature_df(config, dataloader, model)
#             df.to_csv(os.path.join(bf_exp_folder, "feature_data_test.csv"), index=False)
