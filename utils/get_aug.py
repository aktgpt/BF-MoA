import albumentations as aug


def get_aug(aug_type):
    geo_transforms = aug.Compose(
        [
            aug.RandomCrop(1024, 1024, p=1),
            aug.Resize(512, 512, p=1),
            aug.RandomGridShuffle(grid=(3, 3)),
            aug.Flip(),
            aug.RandomRotate90(),
        ]
    )
    color_augs = [
        aug.OneOf(
            [
                aug.GaussianBlur(),
                aug.MotionBlur(),
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
            # p=0.2,
            p=1.0,
        )
    ]
    if "ch_aug" in aug_type:
        colour_transforms = aug.Compose(
            [
                aug.ToFloat(max_value=65535.0),
                # aug.PerChannel(color_augs, p=0.5),
                aug.PerChannel2(color_augs, n_channels_to_aug=1, p=0.5),
                aug.FromFloat(dtype="float64", max_value=65535.0),
            ]
        )
    elif "global_aug" in aug_type:
        colour_transforms = aug.Compose(
            [
                aug.ToFloat(max_value=65535.0),
                aug.Compose(color_augs, p=0.5),
                aug.FromFloat(dtype="float64", max_value=65535.0),
            ]
        )
    else:
        colour_transforms = None

    valid_transforms = aug.Compose(
        [aug.Resize(1080, 1080, p=1.0), aug.CenterCrop(1024, 1024, p=1.0)]
    )
    return geo_transforms, colour_transforms, valid_transforms
