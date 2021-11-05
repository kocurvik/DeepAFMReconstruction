"""
File contains classes and functions used for synthetic data generation and some AD HOC functions used for some experiments
Data generation is controlled by parameters passed to init.
Usage:
    syn = Synthesizer(params)
    syn.generateData()
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os

from tqdm import tqdm

from synth.artifacts import dilate, apply_x_correlated_noise, grad_overshoot_markov, add_linear_skew, add_parabolic_skew
from synth.generator import generate_grid_structure

# params used for SEM simulation
from utils.image import normalize, load_tips_from_pkl, normalize_joint

from multiprocessing import Pool, Process, Queue


class Synthesizer():
    """
    Synthesizer class. For params description, see default params dict in the end of source file
    """
    def __init__(self, out_path, tips_path, resolution=128, num_workers=0, save_every = None,
                 linear_skew_sigma=0.2, parabolic_skew_sigma=0.2, skew_prob=0.5,
                 noise_prob=0.9, noise_alpha_min=0.001, noise_alpha_max=0.01, noise_sigma_min=0.7, noise_sigma_max=0.8,
                 overshoot_prob=0.8, max_overshoot_t=0.5, max_overshoot_mag=0.1, min_p_keep=0.1, max_p_keep=0.9,
                 min_weaken_factor=0.5, max_weaken_factor=0.9):

        self.out_path = out_path

        self.tips = load_tips_from_pkl(tips_path)
        self.tips_keys = list(self.tips.keys())

        self.entries = []
        self.num_workers = num_workers

        self.resolution = resolution

        self.save_every = save_every

        self.linear_skew_sigma = linear_skew_sigma
        self.parabolic_skew_sigma = parabolic_skew_sigma
        self.skew_prob = skew_prob

        self.noise_prob = noise_prob
        self.noise_alpha_min = noise_alpha_min
        self.noise_alpha_max = noise_alpha_max
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_max = noise_sigma_max

        self.overshoot_prob = overshoot_prob
        self.max_overshoot_t = max_overshoot_t
        self.max_overshoot_mag = max_overshoot_mag
        self.min_p_keep = min_p_keep
        self.max_p_keep = max_p_keep
        self.min_weaken_factor = min_weaken_factor
        self.max_weaken_factor = max_weaken_factor

        self.resolution = resolution
        self.i = 0

    def get_random_tip(self):
        tip = np.random.choice(self.tips_keys)
        rot = np.random.randint(0, 3)
        tip_scaled = normalize(self.tips[tip]['data'])
        scale = np.random.uniform(0.5, 5.0)
        tip_scaled *= scale
        for _ in range(0, rot):
            tip_scaled = np.rot90(tip_scaled)

        return tip_scaled

    def add_overshoot(self, image, flip=False):
        t = np.random.uniform(0, 0.5)
        mag = np.random.uniform(0, 0.1)
        p_keep = np.random.uniform(0.1, 0.95)
        p_weaken = np.random.uniform(0.0, 1 - p_keep - 0.05)
        weaken_factor = np.random.uniform(0.5, 0.9)
        image = grad_overshoot_markov(image, t, mag, p_keep, p_weaken, weaken_factor, flip=flip)

        t = np.random.uniform(0, self.max_overshoot_t)
        mag = np.random.uniform(0, self.max_overshoot_mag)
        p_keep = np.random.uniform(self.min_p_keep, self.max_p_keep)
        p_weaken = np.random.uniform(0.0, 1 - p_keep - 0.05)
        weaken_factor = np.random.uniform(self.min_weaken_factor, self.max_weaken_factor)
        image = grad_overshoot_markov(image, t, mag, p_keep, p_weaken, weaken_factor, flip=flip)

        return image

    def add_skew(self, image):
        image = add_linear_skew(image, sigma_a=self.linear_skew_sigma,  sigma_b = self.linear_skew_sigma)
        image = add_parabolic_skew(image, sigma_a=self.parabolic_skew_sigma, sigma_b=self.parabolic_skew_sigma)
        return image

    def add_noise(self, image, flip=False):
        sigma = np.random.uniform(self.noise_sigma_min, self.noise_sigma_max)
        alpha = np.random.uniform(self.noise_alpha_min, self.noise_alpha_max)
        return apply_x_correlated_noise(image, alpha, sigma, flip=flip)

    def apply_artifacts_and_noise(self, image):
        tip = self.get_random_tip()
        image = dilate(image, tip)

        image_l = image
        image_r = image

        if np.random.rand() < self.overshoot_prob:
            image_l = self.add_overshoot(image_l)
            image_r = self.add_overshoot(image_r, flip=True)

        if np.random.rand() < self.skew_prob:
            image_l = self.add_skew(image_l)
            image_r = self.add_skew(image_r)

        if np.random.rand() < self.noise_prob:
            image_l = self.add_noise(image_l)
            image_r = self.add_noise(image_r, flip=True)

        return image_l, image_r

    def save_np(self):
        np.save(self.out_path, np.array(self.entries, dtype=np.float32))

    def generate_entries(self, n):
        for i in range(n):
            entry = self.generate_single_entry()
            self.entries.append(entry)
            # if self.save_every is not None and i % self.save_every == 0:
            #     print("Saving at index ", i)
            #     self.save_np()
        return np.array(self.entries)

    def generate_single_entry(self):
        image = generate_grid_structure(self.resolution, self.resolution)
        image_l, image_r = self.apply_artifacts_and_noise(image)
        image_l, image_r = normalize_joint(image_l, image_r)
        entry = np.stack([image_l, image_r, image], axis=0).astype(np.float32)
        return entry

# class ProcessSynthetizer(Process):
#     def __init__(self, queue, num_items, synthetizer):
#         super(ProcessSynthetizer, self).__init__()
#         self.queue = queue
#         self.num_items = num_items
#         self.synthetizer = synthetizer
#
#     def run(self):
#         for _ in range(self.num_items):
#             entry = self.synthetizer.generate_single_entry()
#             self.queue.put(entry)


def run_mp(synthetizer, num_items, num_workers=8):
    pool = Pool(num_workers)
    num_per_worker = [num_items // num_workers + (num_items % num_workers > i) for i in range(num_workers)]
    print("Workers will generate ", num_per_worker , "items")
    entries = pool.map(synthetizer.generate_entries, num_per_worker)
    entries = np.concatenate(entries, axis=0)
    return entries

if __name__ == '__main__':
    tip_pkl_path = 'D:/Research/data/GEFSEM/synth/res/tips.pkl'
    train_path = 'D:/Research/data/GEFSEM/synth/generated/train.npy'
    val_path = 'D:/Research/data/GEFSEM/synth/generated/val.npy'

    # for num_workers in range(0, 17, 4):
    for num_workers in range(0, 17, 4):
        start_time = time.time()
        syn = Synthesizer(val_path, tip_pkl_path, num_workers=num_workers)
        run_mp(syn, 3000, num_workers)
        print("For {} workers comp took {}".format(num_workers, time.time() - start_time))

    # start_time = time.time()
    # syn = Synthesizer(3000, val_path, tip_pkl_path, num_workers=num_workers)
    # syn.generate_data()
    # print("For {} workers comp took {}".format(num_workers, time.time() - start_time))

