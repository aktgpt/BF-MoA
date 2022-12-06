import albumentations as aug
from albumentations.core.composition import BaseCompose, TransformsSeqType
import random
import typing


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
            p=1.0,
        )
    ]

    colour_transforms = None

    valid_transforms = aug.Compose(
        [aug.Resize(1080, 1080, p=1.0), aug.CenterCrop(1024, 1024, p=1.0)]
    )
    return geo_transforms, colour_transforms, valid_transforms
