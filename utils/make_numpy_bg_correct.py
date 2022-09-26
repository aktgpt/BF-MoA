import glob
import os
import queue
import random
import shutil
import threading
import time
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from genericpath import isfile
from scipy import ndimage, signal


def gkern(sigma=3):
    """Returns a 2D Gaussian kernel array."""
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(4.0 * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    gkern1d = ndimage.filters._gaussian_kernel1d(sigma, 0, lw)[::-1]
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


gaussian_kernel = gkern(sigma=50)
gaussian_kernel_sum = np.sum(gaussian_kernel)
gaussian_kernel = gaussian_kernel[..., np.newaxis].repeat(6, -1)


fileQueue = queue.Queue()


class ThreadedBGCorrect:
    totalFiles = 0
    copyCount = 0
    lock = threading.Lock()

    def __init__(self, file_list):
        self.totalFiles = len(file_list)

        print(str(self.totalFiles) + " files to process.")
        self.threadWorkerCopy(file_list)

    def CopyWorker(self):
        while True:
            fileName = fileQueue.get()
            self.save_bg_correct(fileName)
            fileQueue.task_done()
            with self.lock:
                self.copyCount += 1
                percent = (self.copyCount * 100) / self.totalFiles
                print(str(percent) + " percent copied.")

    def threadWorkerCopy(self, fileNameList):
        for i in range(32):
            t = threading.Thread(target=self.CopyWorker)
            t.daemon = True
            t.start()
        for fileName in fileNameList:
            fileQueue.put((fileName))
        fileQueue.join()

    def save_bg_correct(self, filename):
        image = np.load(filename)
        if image.shape[-1] == 6:
            bg = (
                signal.fftconvolve(
                    np.pad(image, ((200, 200), (200, 200), (0, 0)), mode="reflect"),
                    gaussian_kernel,
                    mode="valid",
                    axes=(0, 1, 2),
                )
                / 6
            )
            bg_corrected = np.round(image - bg).astype(np.int16)
            np.save(os.path.splitext(filename)[0] + "_bg_corrected.npy", bg_corrected)


image_path = (
    "/proj/haste_berzelius/datasets/specs/scratch3-shared/phil/grit_based_numpy_data/bf/"
)

all_files = sorted(glob.glob(image_path + "/**/*.npy", recursive=True))
all_files_filtered = [f for f in all_files if "bg_corrected.npy" not in f]
wrong_files = [f for f in all_files if "bg_corrected_bg_corrected.npy" in f]
for file in wrong_files:
    os.remove(file)

ThreadedBGCorrect(all_files_filtered)
# stats = []

# for file in tqdm.tqdm(all_files_filtered):
#     image = np.load(file)

#     if image.shape[-1] == 6 and not os.path.isfile(file.replace(".npy", "_bg_corrected.npy")):
#         bg1 = (
#             signal.fftconvolve(
#                 np.pad(image, ((200, 200), (200, 200), (0, 0)), mode="reflect"),
#                 gaussian_kernel,
#                 mode="valid",
#                 axes=(0, 1),
#             )
#             / 6
#         )
#         bg_corrected = np.round(image - bg).astype(np.int16)
#         np.save(os.path.splitext(file)[0] + "_bg_corrected.npy", bg_corrected)
#     # stats.append(np.percentile(image, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100], axis=(0, 1,2)))


# x = 1
# np.save("bg_corrected_stats.npy", np.array(stats))


# all_files = sorted(glob.glob(image_path + "/**/*_bg_corrected.npy", recursive=True))
# x = np.load(all_files[0])
# stats = np.load("bg_corrected_stats.npy")[:, :, 0]
# sns.boxplot(data=stats)
# plt.ylim([-500, 500])
