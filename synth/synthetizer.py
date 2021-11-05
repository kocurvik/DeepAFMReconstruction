# !!! Do not import torch here or in any subimpots!
# This commits too much memory even without using and mp version then runs out of RAM very quickly!
# For details see https://stackoverflow.com/questions/64837376/how-to-efficiently-run-multiple-pytorch-processes-models-at-once-traceback
import json
import os
import time
from datetime import datetime

import git
import numpy as np

from synth.artifacts import dilate, apply_x_correlated_noise, grad_overshoot_markov, add_linear_skew, add_parabolic_skew
from synth.generator import generate_grid_structure

from utils.image import normalize, load_tips_from_pkl, normalize_joint

from multiprocessing import Pool


class Synthesizer():
    """
    Synthesizer class. For params description, see default params dict in the end of source file
    """

    def __init__(self, tips_path, **kwargs):

        self.tips = load_tips_from_pkl(tips_path)
        self.tips_keys = list(self.tips.keys())

        self.entries = []

        default_params = {
            'resolution': 128,
            # skew params
            'linear_skew_sigma': 0.2, 'parabolic_skew_sigma': 0.2, 'skew_prob': 0.5,

            # overshoot params
            'overshoot_prob': 0.8, 'max_overshoot_t': 0.5, 'max_overshoot_mag': 0.1, 'min_p_keep': 0.1,
            'max_p_keep': 0.9, 'min_weaken_factor': 0.5, 'max_weaken_factor': 0.9,

            # x-correlated noise params
            'noise_prob': 0.9, 'noise_alpha_min': 0.001, 'noise_alpha_max': 0.01, 'noise_sigma_min': 0.7,
            'noise_sigma_max': 0.8,

            # dilation params
            'tip_scale_min': 0.5, 'tip_scale_max': 5.0}

        # for hashing
        self.param_names = sorted(list(default_params.keys()))

        for (prop, default) in default_params.items():
            setattr(self, prop, kwargs.get(prop, default))

    def get_param_dict(self):
        return {param_name: getattr(self, param_name) for param_name in self.param_names}

    def __hash__(self):
        return hash(json.dumps({param_name: getattr(self, param_name) for param_name in self.param_names}, sort_keys=True))

    def get_random_tip(self):
        tip = np.random.choice(self.tips_keys)
        rot = np.random.randint(0, 3)
        tip_scaled = normalize(self.tips[tip]['data'])
        scale = np.random.uniform(self.tip_scale_min, self.tip_scale_max)
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
        image = add_linear_skew(image, sigma_a=self.linear_skew_sigma, sigma_b=self.linear_skew_sigma)
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

    # def save_np(self):
    #     np.save(self.out_path, np.array(self.entries, dtype=np.float32))

    def generate_entries(self, n):
        for i in range(n):
            entry = self.generate_single_entry()
            self.entries.append(entry)
        return np.array(self.entries)

    def generate_single_entry(self):
        image = generate_grid_structure(self.resolution, self.resolution)
        image_l, image_r = self.apply_artifacts_and_noise(image)
        image_l, image_r = normalize_joint(image_l, image_r)
        entry = np.stack([image_l, image_r, image], axis=0).astype(np.float32)
        return entry


def generate_dataset(synthetizer, num_items, num_workers=8):
    if num_workers > 1:
        pool = Pool(num_workers)
        num_per_worker = [num_items // num_workers + (num_items % num_workers > i) for i in range(num_workers)]
        print("Workers will generate ", num_per_worker, "items")
        entries = pool.map(synthetizer.generate_entries, num_per_worker)
        entries = np.concatenate(entries, axis=0)
        return entries
    else:
        print("Single worker will generete {} items".format(num_items))
        entries = synthetizer.generate_entries(num_items)
        return entries


def save_dataset(synthetizer, entries, dir_path, subset):
    hash_path = os.path.join(dir_path, str(abs(hash(synthetizer))))

    if not os.path.isdir(hash_path):
        os.mkdir(hash_path)

    json_path = os.path.join(dir_path, hash_path, '{}.json'.format(subset))
    npy_path = os.path.join(dir_path, hash_path, '{}.npy'.format(subset))

    param_dict = synthetizer.get_param_dict()
    repo = git.Repo(search_parent_directories=True)

    json_dict = {'date': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'), 'num_items': len(entries),
                 'synthetizer_params': param_dict, 'git_hash': repo.head.object.hexsha}

    np.save(npy_path, entries)

    with open(json_path, 'w') as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    tip_pkl_path = 'D:/Research/data/GEFSEM/synth/res/tips.pkl'
    out_path = 'D:/Research/data/GEFSEM/synth/generated/'

    syn = Synthesizer(tip_pkl_path)
    entries = generate_dataset(syn, num_items=4000, num_workers=8)
    save_dataset(syn, entries, out_path, 'train')

    entries = generate_dataset(syn, num_items=1000, num_workers=8)
    save_dataset(syn, entries, out_path, 'val')

    # for num_workers in range(0, 17, 4):
    # for num_workers in range(0, 17, 4):
    #     start_time = time.time()
    #     syn = Synthesizer(tip_pkl_path, num_workers=num_workers)
    #     generate_dataset(syn, 3000, num_workers)
    #     print("For {} workers comp took {}".format(num_workers, time.time() - start_time))
