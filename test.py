from data.bf_dataset import BFDataset
from torch.utils.data import DataLoader
import psutil
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import albumentations as aug
from tqdm import tqdm
import matplotlib

# matplotlib.use("Qt5Agg")

moas = [
    "Aurora kinase inhibitor",
    "tubulin polymerization inhibitor",
    # "JAK inhibitor",
    "protein synthesis inhibitor",
    "HDAC inhibitor",
    "topoisomerase inhibitor",
    "PARP inhibitor",
    "ATPase inhibitor",
    "retinoid receptor agonist",
    "HSP inhibitor",
    "dmso",
]

geo_transforms = aug.Compose(
    [
        aug.RandomCrop(1024, 1024),
        aug.Resize(512, 512),
        aug.RandomGridShuffle(grid=(3, 3)),
        aug.Flip(),
        aug.RandomRotate90(),
    ]
)

colour_transforms = aug.Compose(
    [
        aug.ToFloat(max_value=65535.0),
        aug.PerChannel(
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
                p=0.3,
            ),
            p=0.5,
        ),
        aug.FromFloat(max_value=65535.0),
    ]
)
#  aug.PerChannel(
#     aug.OneOf(
#         [
#             aug.GaussianBlur(),
#             aug.MotionBlur(),
#             aug.MedianBlur(blur_limit=5),
#             aug.GaussNoise(var_limit=(0.1, 1.0)),
#             aug.CoarseDropout(
#                 max_holes=32, max_height=32, max_width=32, min_height=16, min_width=16
#             ),
#         ],
#         p=0.3,
#     ),
#     p=0.5,
# )
dmso_stats_path = "stats/final_stats/bf_stats_tif.csv"  # "stats/bf_dmso_stats.csv"#   #"stats/new_stats/bf_dmso_stats.csv"

train_dataset = BFDataset(
        root="/proj/haste_berzelius/datasets/specs",
        csv_file="stats/new_train_val_test_splits/bf_subset_train_split2.csv",
        normalize="comp",
        dmso_stats_path=dmso_stats_path,
        moas=moas,
        geo_transform=geo_transforms,
        colour_transform=colour_transforms,
    )


stats = []
for i in tqdm(range(len(train_dataset))):
    sample = train_dataset[i]
    x = sample[0]
    print(
        np.percentile(x, [0, 25, 50, 75, 100]),
        sample[1],
        sample[2],
        sample[3],
        sample[4],
        sample[5],
    )
    # np.histogram(x, bins=100)
    stats.append(np.percentile(x, [0, 25, 50, 75, 100]))
    img = x[:, :, :3]
    # plt.imshow((img - (-5)) / (5 - (-5)))
    # # plt.imshow((img - np.min(img)) / (np.max(img) - np.min(img)))
    # # clim=(np.percentile(x[:3], 0), np.percentile(x[:3], 100)),
    # # )
    # plt.show()
x_32 = train_dataset.__getitem__(0, "float32")[0]
x_64 = train_dataset.__getitem__(0, "float64")[0]


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=8,
    prefetch_factor=8,
    persistent_workers=True,
)
process = psutil.Process()

mem_used = []
mem_used.append(psutil.virtual_memory().used / 1024 ** 3)
print(process.memory_info().vms / 1024 ** 3)
start = time.time()
for i, item in enumerate(train_loader):
    if i % 1 == 0:
        print(process.memory_info().vms / 1024 ** 3)
        mem = psutil.virtual_memory()
        print(
            f"{(time.time()-start)/(i+1):2.2f} - {i:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}"
        )
        mem_used.append(mem.used / 1024 ** 3)

plt.plot(np.array(mem_used))
plt.show()
