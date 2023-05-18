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
from scipy import ndimage, signal, stats
from skimage import restoration, feature, filters, util, morphology


def gkern(sigma=3):
    """Returns a 2D Gaussian kernel array."""
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(4.0 * sd + 0.1)
    # Since we are calling correlate, not convolve, revert the kernel
    gkern1d = ndimage.filters._gaussian_kernel1d(sigma, 0, lw)[::-1]
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def standardize(image, min, max):
    return (image - min) / (max - min)


import numpy as np
from skimage.restoration import denoise_tv_bregman
from skimage.util import img_as_float
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def _tv_restore(image, lmbda):
    w = 1 / lmbda**2
    return denoise_tv_bregman(image / np.sqrt(w)) * np.sqrt(w)


def _objective_function(x, y, forward, forward_t, lmbda):
    return np.linalg.norm(y - x) ** 2 + lmbda * np.linalg.norm(forward(x)) ** 2


def _gradient(x, y, forward, forward_t, lmbda):
    return 2 * (x - y) + 2 * lmbda * forward_t((forward(x)))


def _create_finite_diff(shape):
    Dy = (np.eye(shape[0], k=1) - np.eye(shape[0], k=0)).T
    Dx = np.eye(shape[1], k=1) - np.eye(shape[1], k=0)
    forward = lambda x: (Dy @ x.reshape(shape) @ Dx).ravel()
    forward_t = lambda x: (Dy.T @ x.reshape(shape) @ Dx.T).ravel()
    return forward, forward_t


def _l2_restore(Y, lmbda):
    forward, forward_t = _create_finite_diff(Y.shape)
    y = Y.ravel()
    x0 = np.zeros(y.shape[0])
    result = minimize(
        _objective_function,
        x0,
        jac=_gradient,
        args=(y, forward, forward_t, lmbda),
        method="Newton-CG",
        tol=1e-6 * lmbda,
    )
    return result.x.reshape(Y.shape)


def gupta_restoration(image: np.ndarray, smoothness_reg: float = 1000000.0, tv_reg: float = 1.0):
    # Force float
    image = img_as_float(image)
    background = np.zeros_like(image)
    foreground = np.zeros_like(image)
    for _ in tqdm(range(30)):
        # Solve background
        background = _l2_restore(image - foreground, smoothness_reg)
        foreground = _tv_restore(image - background, tv_reg)
    return foreground, background


# EXAMPLE
# image = np.random.rand(256, 256)
# foreground, background = gupta_restoration(image)
# plt.figure()
# plt.imshow(foreground)
# plt.figure()
# plt.imshow(background)
# plt.show()


fileQueue = queue.Queue()


class ThreadedBGCorrect:
    totalFiles = 0
    copyCount = 0
    lock = threading.Lock()

    def __init__(
        self, file_list, kernel_size, per_channel=False, noise_filter=None, denoise_first=True
    ):
        self.file_list = file_list
        self.totalFiles = len(file_list)

        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.gaussian_kernel = gkern2(kernlen=kernel_size)[..., np.newaxis].repeat(5, -1)
            self.gaussian_kernel_sum = np.sum(self.gaussian_kernel)
            self.per_channel = per_channel
            self.pad_x, self.pad_y = int(kernel_size // 2), int(kernel_size // 2)

        self.noise_filter = noise_filter
        self.denoise_first = denoise_first

        print(str(self.totalFiles) + " files to process.", flush=True)
        # self.threadWorkerCopy(file_list)

    def CopyWorker(self):
        while True:
            fileName = fileQueue.get()
            self.save_bg_correct(fileName)
            fileQueue.task_done()
            with self.lock:
                self.copyCount += 1
                percent = (self.copyCount * 100) / self.totalFiles
                print(str(percent) + " percent copied.", flush=True)

    def threadWorkerCopy(self):
        for i in range(16):
            t = threading.Thread(target=self.CopyWorker)
            t.daemon = True
            t.start()
        for fileName in self.file_list:
            fileQueue.put((fileName))
        fileQueue.join()

    def get_bg_correct(self, image):
        image_padded = np.pad(
            image,
            ((self.pad_x, self.pad_x), (self.pad_y, self.pad_y), (0, 0)),
            mode="reflect",
        )
        if self.per_channel:
            bg = signal.fftconvolve(
                image_padded,
                self.gaussian_kernel[:, :, 0][:, :, np.newaxis],
                mode="valid",
                axes=(0, 1, 2),
            )
        else:
            bg = (
                signal.fftconvolve(
                    image_padded,
                    self.gaussian_kernel,
                    mode="valid",
                    axes=(0, 1, 2),
                )
                / self.gaussian_kernel_sum
            )
        bg_corrected = image - bg  # np.round(image - bg).astype(np.int16)
        assert bg_corrected.shape == image.shape
        return bg_corrected

    def get_denoised(self, image, min, max):
        image_min = np.percentile(image, min)  # image.min()
        image_max = np.percentile(image, max)  # image.max()
        image = (image - image_min) / (image_max - image_min)
        # image = np.clip(image, 0, 1)
        if "med" in self.noise_filter:
            image_filtered = ndimage.median_filter(image, size=(5, 5, 1))
        elif "tv" in self.noise_filter:
            image_filtered = restoration.denoise_tv_chambolle(image, weight=0.1, channel_axis=-1)
        image_filtered = (image_filtered * (image_max - image_min)) + image_min
        return image_filtered

    def get_image(self, image, min=0.5, max=99.5):
        out_image = image.copy()
        if self.denoise_first:
            if self.noise_filter is not None:
                out_image = self.get_denoised(out_image, min=min, max=max)
            if self.kernel_size is not None:
                out_image = self.get_bg_correct(out_image)
        else:
            if self.kernel_size is not None:
                out_image = self.get_bg_correct(out_image)
            if self.noise_filter is not None:
                out_image = self.get_denoised(out_image, min=min, max=max)
        out_image = out_image.astype(np.int16)
        return out_image

    def get_name(self, filename):
        save_name = os.path.splitext(filename)[0]
        if self.denoise_first:
            if self.noise_filter is not None:
                save_name = save_name + f"_{self.noise_filter}"
            if self.kernel_size is not None:
                save_name = save_name + f"_bg_corrected_{self.kernel_size}"
                save_name = save_name + f"_pc" if self.per_channel else save_name
        else:
            if self.kernel_size is not None:
                save_name = save_name + f"_bg_corrected_{self.kernel_size}"
                save_name = save_name + f"_pc" if self.per_channel else save_name
            if self.noise_filter is not None:
                save_name = save_name + f"_{self.noise_filter}"
        return save_name

    def save_bg_correct(self, filename):
        image = np.load(filename)

        out_image = self.get_image(image)
        save_name = self.get_name(filename)

        np.save(save_name + ".npy", out_image)

        # out_image = image.copy()
        # save_name = os.path.splitext(filename)[0]

        # if image.shape[-1] == 6:
        #     if self.denoise_first:
        #         if self.noise_filter is not None:
        #             out_image = self.get_denoised(out_image)
        #             save_name = save_name + f"_{self.noise_filter}"
        #         if self.kernel_size is not None:
        #             out_image = self.get_bg_correct(out_image)
        #             save_name = save_name + f"_bg_corrected_{self.kernel_size}"
        #             save_name = save_name + f"_pc" if self.per_channel else save_name
        #     else:
        #         if self.kernel_size is not None:
        #             out_image = self.get_bg_correct(out_image)
        #             save_name = save_name + f"_bg_corrected_{self.kernel_size}"
        #             save_name = save_name + f"_pc" if self.per_channel else save_name
        #         if self.noise_filter is not None:
        #             out_image = self.get_denoised(out_image)
        #             save_name = save_name + f"_{self.noise_filter}"

        #     out_image = out_image.astype(np.int16)


image_path = (
    "/proj/haste_berzelius/datasets/specs/scratch3-shared/phil/non_grit_based_numpy_data/fl/"
    # "/proj/haste_berzelius/datasets/AG-48h-P3-L2/scratch2-shared/phil/bf_moa_live/AG-48h-P3-L2_final_time_point/numpy_data/bf/"
)

all_files = sorted(glob.glob(image_path + "/**/*.npy", recursive=True))
all_files_filtered = [f for f in all_files if "bg_corrected" not in f]

# wrong_files = [f for f in all_files if "_med_bg_corrected_301" in f]
# for file in wrong_files:
#    os.remove(file)

bg_correct_worker = ThreadedBGCorrect(
    all_files_filtered, kernel_size=101, per_channel=True, noise_filter=None, denoise_first=False
)
bg_correct_worker.threadWorkerCopy()

# bg_correct_worker1 = ThreadedBGCorrect(
#     all_files_filtered, kernel_size=101, per_channel=True, noise_filter="med", denoise_first=True
# )

# for i in range(len(all_files_filtered)):
#     image = np.load(all_files_filtered[i + 100])

#     image_bg_correct = bg_correct_worker.get_bg_correct(image)

#     image_min = np.percentile(image_bg_correct, 0.1)
#     image_max = np.percentile(image_bg_correct, 99.9)

#     image_denoise_first = bg_correct_worker.get_image(image, min=1, max=99)
#     image_bg_correct_first = bg_correct_worker1.get_image(image, min=0.5, max=99.5)

#     fig, ax = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True, tight_layout=True)
#     ax[0].imshow(
#         standardize(image[:, :, :1], min=np.percentile(image, 0.1), max=np.percentile(image, 99.9)),
#         cmap="gray",
#     )
#     ax[1].imshow(standardize(image_bg_correct, image_min, image_max)[:, :, :1], cmap="gray")
#     ax[2].imshow(standardize(image_denoise_first, image_min, image_max)[:, :, :1], cmap="gray")
#     ax[3].imshow(standardize(image_bg_correct_first, image_min, image_max)[:, :, :1], cmap="gray")
#     plt.show()


# kernel_sizes = [101, 301]
# all_kernel_files = []
# for kernel_size in kernel_sizes:
#     kernel_files = sorted([f for f in all_files if f"_bg_corrected_{kernel_size}.npy" in f])
#     # kernel_files = [f for f in kernel_files if "med" not in f]
#     # kernel_files = [f for f in kernel_files if "tv" not in f]
#     all_kernel_files.append(kernel_files)

# ## view the images side by side with original and zoom in together
# for i in range(len(all_files_filtered)):
#     image = np.load(all_files_filtered[i + 5010])
#     image_min = np.percentile(image, 0.1)
#     image_max = np.percentile(image, 99.9)

#     # foreground, background = gupta_restoration(image[:,:,0])

#     # image = (image - image_min) / (image_max - image_min)
#     # tv_denoised = restoration.denoise_tv_chambolle(image, weight=0.05, channel_axis=-1)
#     # tv_denoised = bg_correct_worker.get_bg_correct(tv_denoised)[:, :, :3]
#     # med_denoised = ndimage.median_filter(image, size=(1, 1, 6))
#     # med_denoised = bg_correct_worker.get_bg_correct(med_denoised)[:, :, :3]
#     # image = bg_correct_worker.get_bg_correct(image)[:, :, :3]
#     # edge_image = filters.sobel(image)

#     image_101 = np.load(all_kernel_files[0][i + 5010])[:, :, :3]
#     image_101_min = np.percentile(image_101, 0.1)
#     image_101_max = np.percentile(image_101, 99.9)
#     image_101 = (image_101 - image_101_min) / (image_101_max - image_101_min)
#     tv_denoised_101 = restoration.denoise_tv_chambolle(image_101, weight=0.05, channel_axis=-1)
#     med_denoised_101 = ndimage.median_filter(image_101, size=(2, 2, 6))
#     edge_image_101 = filters.sobel(image_101)  # [:, :, :3]

#     rand_img = np.random.rand(*image_101.shape)
#     image_salted = rand_img > 1 - 0.0001 / 2
#     image_peppered = rand_img < 0.0001 / 2

#     footprint = np.ones((5, 5, 3), dtype=bool)
#     footprint[np.random.rand(5, 5, 3) < 0.5] = 0

#     img_salted = morphology.binary_dilation(image_salted, footprint=footprint).astype(float)
#     img_peppered = morphology.binary_dilation(image_peppered, footprint=footprint).astype(float)
#     img_sp = 1 + img_salted - img_peppered
#     image_noisy_101 = image_101 * img_sp

#     x_noisy = (util.random_noise(x, mode="s&p", amount=0.0001) * 255).astype(np.uint8)
#     footprint = np.ones((5, 5, 2))
#     footprint[np.random.rand(5, 5, 2) < 0.4] = 0
#     x_salted = filters.rank.maximum(x_noisy, footprint)
#     x_salted1 = (x_salted - 127) / 128
#     x_peppered = filters.rank.minimum(x_noisy, footprint)
#     x_peppered1 = (x_peppered.astype(float) - 127) / 128
#     x_noisy_filtered = x_salted1 + x_peppered1
#     image_noisy_101 = image_101 + (image_101 * x_noisy_filtered)

#     image_301 = np.load(all_kernel_files[1][i + 5010])[:, :, :3]
#     image_301_min = np.percentile(image_301, 0.1)
#     image_301_max = np.percentile(image_301, 99.9)
#     image_301 = (image_301 - image_301_min) / (image_301_max - image_301_min)
#     tv_denoised_301 = restoration.denoise_tv_chambolle(image_301, weight=0.05, channel_axis=-1)
#     med_denoised_301 = ndimage.median_filter(image_301, size=(5, 5, 1))

#     fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True, tight_layout=True)
#     for i in range(5):
#         # ax[i].set_aspect("equal")
#         ax[i].axis("off")
#     ax[0].imshow(edge_image.sum(-1))
#     ax[1].imshow(tv_denoised)
#     ax[2].imshow(np.abs(image - tv_denoised), interpolation="none")
#     ax[3].imshow(med_denoised)
#     ax[4].imshow(np.abs(image - med_denoised), interpolation="none")
#     plt.subplots_adjust(wspace=0, hspace=0)

#     fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True, tight_layout=True)
#     for i in range(5):
#         ax[i].set_aspect("equal")
#         ax[i].axis("off")
#     ax[0].imshow(image_noisy_101)
#     ax[1].imshow(tv_denoised_101)
#     ax[2].imshow(np.abs(image_101 - tv_denoised_101), interpolation="none")
#     ax[3].imshow(med_denoised_101)
#     ax[4].imshow(np.abs(image_101 - med_denoised_101), interpolation="none")
#     plt.subplots_adjust(wspace=0, hspace=0)

#     fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True, tight_layout=True)
#     for i in range(5):
#         ax[i].set_aspect("equal")
#         ax[i].axis("off")
#     ax[0].imshow(image_301)
#     ax[1].imshow(tv_denoised_301)
#     ax[2].imshow(np.abs(image_301 - tv_denoised_301), interpolation="none")
#     ax[3].imshow(med_denoised_301)
#     ax[4].imshow(np.abs(image_301 - med_denoised_301), interpolation="none")
#     plt.subplots_adjust(wspace=0, hspace=0)

#     plt.show()

#     # image_min = image.min()
#     # image_max = image.max()

#     # image = signal.medfilt2d(image, kernel_size=3)
#     # # image = (image - image_min) / (image_max - image_min)
#     # image = (image - np.percentile(image, 0.1)) / (
#     #     np.percentile(image, 99.9) - np.percentile(image, 0.1)
#     # )

#     # image_101 = np.load(all_kernel_files[0][i])[:, :, :3]
#     # image_101 = signal.medfilt2d(image_101, kernel_size=3)
#     # # image_101 = (image_101 - image_101.min()) / (image_101.max() - image_101.min())
#     # image_101 = (image_101 - np.percentile(image_101, 0.1)) / (
#     #     np.percentile(image_101, 99.9) - np.percentile(image_101, 0.1)
#     # )
#     # image_201 = np.load(all_kernel_files[1][i])[:, :, :3]
#     # image_201 = signal.medfilt2d(image_201, kernel_size=3)
#     # # image_201 = (image_201 - image_201.min()) / (image_201.max() - image_201.min())
#     # image_201 = (image_201 - np.percentile(image_201, 0.1)) / (
#     #     np.percentile(image_201, 99.9) - np.percentile(image_201, 0.1)
#     # )

#     # image_301 = np.load(all_kernel_files[2][i])[:, :, :3]
#     # image_301 = signal.medfilt2d(image_301, kernel_size=3)
#     # # image_301 = (image_301 - image_301.min()) / (image_301.max() - image_301.min())
#     # image_301 = (image_301 - np.percentile(image_301, 0.1)) / (
#     #     np.percentile(image_301, 99.9) - np.percentile(image_301, 0.1)
#     # )

#     # image_401 = np.load(all_kernel_files[3][i])[:, :, :3]
#     # image_401 = signal.medfilt2d(image_401, kernel_size=3)
#     # image_401 = (image_401 - np.percentile(image_401, 0.1)) / (
#     #     np.percentile(image_401, 99.9) - np.percentile(image_401, 0.1)
#     # )
#     # # image_401 = (image_401 - image_401.min()) / (image_401.max() - image_401.min())
#     # fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True, tight_layout=True)
#     # ax[0].imshow(image)
#     # ax[0].set_title("original")
#     # ax[1].imshow(image_101)
#     # ax[1].set_title("kernel size 101")
#     # ax[2].imshow(image_201)
#     # ax[2].set_title("kernel size 201")
#     # ax[ 0].imshow(image_301)
#     # ax[ 0].set_title("kernel size 301")
#     # ax[ 1].imshow(image_401)
#     # ax[ 1].set_title("kernel size 401")
#     # plt.show()
#     ## plot the histograms
#     # fig, ax = plt.subplots(2, 3, figsize=(15, 10))#, sharex=True, sharey=True, tight_layout=True)
#     # ax[0].hist(image.flatten(), bins=100)
#     # ax[0].set_title("original")
#     # ax[1].hist(image_101.flatten(), bins=100)
#     # ax[1].set_title("kernel size 101")
#     # ax[2].hist(image_201.flatten(), bins=100)
#     # ax[2].set_title("kernel size 201")
#     # ax[ 0].hist(image_301.flatten(), bins=100)
#     # ax[ 0].set_title("kernel size 301")
#     # ax[ 1].hist(image_401.flatten(), bins=100)
#     # ax[ 1].set_title("kernel size 401")
#     plt.show()


# wrong_files = sorted(
#     [f for f in all_files if "_bg_corrected_pc_101.npy" in f]
# )  # or "bg_corrected_pc.npy"
# # # wrong_files = [f for f in all_files if "bg_corrected_pc.npy" in f]
# # for file in wrong_files:
# #     os.remove(file)
# # for kernel_size in [
# #     101,
# # ]:
# ThreadedBGCorrect(all_files_filtered, kernel_size=401, per_channel=False)
