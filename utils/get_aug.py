import random
import typing
from typing import Any, Dict

import albumentations as aug
import numpy as np
from albumentations.core.composition import BaseCompose, TransformsSeqType
from albumentations.core.transforms_interface import ImageOnlyTransform
from skimage import morphology


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
    sp_augs = aug.OneOf(
        [
            PerChannel2(aug.CoarseDropout(p=1), n_channels_to_aug=1, p=0.5),
            SPTransform(channels=(1, 6), radius=(1, 4), sp_prob=0.0005, p=0.5),
        ],
        p=1.0,
    )

    if "ch_aug" in aug_type:
        colour_transforms = aug.Compose(
            [
                aug.ToFloat(max_value=65535.0),
                PerChannel2(color_augs, n_channels_to_aug=1, p=0.5),
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
    elif "sp_aug" in aug_type:
        colour_transforms = aug.Compose(
            [
                aug.ToFloat(max_value=65535.0, always_apply=True),
                aug.Compose(sp_augs, p=0.5),
                aug.FromFloat(dtype="float64", max_value=65535.0, always_apply=True),
            ]
        )
    else:
        colour_transforms = None

    valid_transforms = aug.Compose(
        [aug.Resize(1080, 1080, p=1.0), aug.CenterCrop(1024, 1024, p=1.0)]
    )
    return geo_transforms, colour_transforms, valid_transforms


class SPTransform(ImageOnlyTransform):
    def __init__(self, channels=(1, 3), radius=(3, 9), sp_prob=0.0001, always_apply=False, p=0.5):
        super(SPTransform, self).__init__(always_apply, p)
        self.channels = channels
        self.radius = radius
        self.sp_prob = sp_prob

    def apply(self, img, channels, radius, **params):
        footprint = morphology.disk(radius, dtype=bool)
        footprint = np.repeat(footprint[:, :, np.newaxis], channels, axis=2)
        footprint[np.random.rand(*footprint.shape) < 0.25] = 0

        rand_img = np.random.rand(*img.shape)

        img_salted = rand_img > (1 - (self.sp_prob / 2))
        img_salted = morphology.binary_dilation(img_salted, footprint=footprint).astype(float)

        img_peppered = rand_img < (self.sp_prob / 2)
        img_peppered = morphology.binary_dilation(img_peppered, footprint=footprint).astype(float)

        img_max = np.percentile(img, 99.99)
        img_min = np.percentile(img, 0.01)

        transformed_img = img.copy()
        transformed_img[img_salted > 0] = img_max
        transformed_img[img_peppered > 0] = img_min

        return transformed_img

    def get_params(self):
        # Select random values for channels and kernel_size
        params = {
            "channels": np.random.randint(self.channels[0], self.channels[1] + 1),
            "radius": np.random.randint(self.radius[0], self.radius[1] + 1),
        }
        return params

    def get_transform_init_args_names(self):
        return ("channels", "sp_prob", "radius")


class CornerCrop(ImageOnlyTransform):
    def __init__(self, height, width, always_apply: bool = False, p: float = 0.5):
        super(CornerCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, corner, **params):
        if corner == "top_left":
            x_start, x_end = 0, self.width
            y_start, y_end = 0, self.height
        elif corner == "top_right":
            x_start, x_end = img.shape[1] - self.width, img.shape[1]
            y_start, y_end = 0, self.height
        elif corner == "bottom_left":
            x_start, x_end = 0, self.width
            y_start, y_end = img.shape[0] - self.height, img.shape[0]
        elif corner == "bottom_right":
            x_start, x_end = img.shape[1] - self.width, img.shape[1]
            y_start, y_end = img.shape[0] - self.height, img.shape[0]

        transformed_img = img[y_start:y_end, x_start:x_end, :]
        return transformed_img

    def get_params(self):
        params = {"corner": random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])}
        return params

    def get_transform_init_args_names(self):
        return ("height", "width")


class PerChannel2(BaseCompose):
    """Apply transformations per-channel

    Args:
        transforms (list): list of transformations to compose.
        channels (sequence): channels to apply the transform to. Pass None to apply to all.
                         Default: None (apply to all)
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        channels: typing.Optional[typing.Sequence[int]] = None,
        n_channels_to_aug: int = 1,
        p: float = 0.5,
    ):
        super(PerChannel2, self).__init__(transforms, p)
        self.channels = channels
        self.n_channels_to_aug = n_channels_to_aug

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
        if force_apply or random.random() < self.p:
            image = data["image"]

            self.channels = range(image.shape[2])
            aug_c = random.sample(list(self.channels), self.n_channels_to_aug)
            # aug_c = np.random.choice(
            #     list(self.channels),
            #     self.n_channels_to_aug,
            #     replace=False,
            #     p=[0.25, 0.2, 0.05, 0.05, 0.2, 0.25],
            # )
            for c in aug_c:
                for t in self.transforms:
                    image[:, :, c] = t(image=image[:, :, c])["image"]
                    # print("augmented channel", c)
            data["image"] = image

        return data
