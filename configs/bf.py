import albumentations as aug

dataset_config = 1


config = dict(
    exp_folder="/proj/haste_berzelius/exps/specs_new_splits",
    exp_name="bf_exps_1_split1",
    exp_mode="bf_11cls_chaug_1000e_sgd_bg_correct",
    data=dict(
        batch_size=32,
        train_csv_path="stats/new_train_val_test_splits/bf_subset_train_split1.csv",
        val_csv_path="stats/new_train_val_test_splits/bf_subset_val_split1.csv",
        test_csv_path="stats/new_train_val_test_splits/bf_subset_test_split1.csv",
        moas=[
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
        ],
        dmso_stats_path="stats/final_stats/all_bf_stats_tif.csv",
        normalization="dmso",
        bg_correct=False,
        train_aug=dict(
            geo_transform=aug.Compose(
                [
                    aug.RandomCrop(1024, 1024),
                    aug.Resize(512, 512),
                    aug.RandomGridShuffle(grid=(3, 3)),
                    aug.Flip(),
                    aug.RandomRotate90(),
                ]
            ),
            colour_transform=aug.Compose(
                [
                    aug.ToFloat(max_value=65535.0),
                    aug.PerChannel(
                        [
                            aug.OneOf(
                                [
                                    aug.GaussianBlur(),
                                    aug.MotionBlur(),
                                    aug.RandomBrightnessContrast(
                                        brightness_limit=0.1,
                                        contrast_limit=0.1,
                                        brightness_by_max=False,
                                    ),
                                    aug.MedianBlur(blur_limit=3),
                                    aug.GaussNoise(var_limit=(0.001, 0.005)),
                                    aug.CoarseDropout(
                                        max_holes=32,
                                        max_height=32,
                                        max_width=32,
                                        min_height=16,
                                        min_width=16,
                                    ),
                                ],
                                p=0.5,
                            )
                        ],
                        p=0.75,
                    ),
                    aug.FromFloat(dtype="float64", max_value=65535.0),
                ]
            ),
            valid_transforms=aug.Compose(
                [aug.Resize(1080, 1080, p=1.0), aug.CenterCrop(1024, 1024, p=1.0)]
            ),
        ),
    ),
    train=dict(
        balanced=True,
        lr_schedule=True,
        batch_size=64,
        trainer="BaseDistTrainer",
        ckpt_path=False,
        epochs=1000,
        init_lr=0.05,
        wd=1e-4,
    ),
    test=dict(
        tester="BaseTester",
        model_path="model_best_accuracy.pth",
    ),
    model=dict(
        type="ResNet",
        args=dict(
            in_channels=6,
            model_name="resnet50",
            num_classes=11,
        ),
    ),
)
