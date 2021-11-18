# !!! Do not import torch here or in any subimpots!
# This commits too much memory even without using and mp version then runs out of RAM very quickly!
# For details see https://stackoverflow.com/questions/64837376/how-to-efficiently-run-multiple-pytorch-processes-models-at-once-traceback
import argparse
import json
import os
import time
from datetime import datetime
from hashlib import sha1

import git
import numpy as np

from synth.artifacts import dilate, apply_x_correlated_noise, grad_overshoot_markov, add_linear_skew, \
    add_parabolic_skew, faster_dilate
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

        default_params = self.get_default_param_dict()
        # for hashing
        self.param_names = sorted(list(default_params.keys()))

        for (prop, default) in default_params.items():
            setattr(self, prop, kwargs.get(prop, default))

    @staticmethod
    def get_default_param_dict():
        default_params = {
            'resolution': 128,
            # skew params
            'linear_skew_sigma': 0.1, 'parabolic_skew_sigma': 0.1, 'skew_prob': 0.25,

            # overshoot params
            'overshoot_prob': 0.7, 'max_overshoot_t': 0.5, 'max_overshoot_mag': 0.05, 'min_p_keep': 0.0,
            'max_p_keep': 0.7, 'min_weaken_factor': 0.0, 'max_weaken_factor': 0.5,

            # x-correlated noise params
            'noise_prob': 0.95, 'noise_alpha_min': 0.00, 'noise_alpha_max': 0.9, 'noise_sigma_min': 0.0,
            'noise_sigma_max': 0.03,

            # dilation params
            'tip_scale_min': 1.0, 'tip_scale_max': 10.0}

        return default_params

    def get_param_dict(self):
        return {param_name: getattr(self, param_name) for param_name in self.param_names}

    def get_param_hash(self):
        dict = {param_name: getattr(self, param_name) for param_name in self.param_names}
        json_string = json.dumps(dict, sort_keys=True)
        return sha1(json_string.encode('utf-8')).hexdigest()

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
        image = faster_dilate(image, tip)

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

    def generate_entries(self, n):
        for i in range(n):
            entry = self.generate_single_entry()
            self.entries.append(entry)
        return np.array(self.entries)

    def generate_single_entry(self):
        image = generate_grid_structure(self.resolution, self.resolution)
        image_l, image_r = self.apply_artifacts_and_noise(image)
        images = normalize_joint([image_l, image_r, image])
        entry = np.stack(images, axis=0).astype(np.float32)
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


def save_dataset(synthetizer, entries, args):
    hash_path = os.path.join(args.out_path, synthetizer.get_param_hash())

    print('Hash path: ', hash_path)

    if not os.path.isdir(hash_path):
        os.mkdir(hash_path)

    json_path = os.path.join(args.out_path, hash_path, '{}.json'.format(args.subset))
    npy_path = os.path.join(args.out_path, hash_path, '{}.npy'.format(args.subset))

    param_dict = synthetizer.get_param_dict()
    repo = git.Repo(search_parent_directories=True)

    json_dict = {'date': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'), 'num_items': len(entries),
                 'synthetizer_params': param_dict, 'git_hash': repo.head.object.hexsha, 'args': vars(args)}

    np.save(npy_path, entries)
    print('Saved to ', npy_path)

    with open(json_path, 'w') as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)

    print('Saved to ', json_path)


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()

    default_params = Synthesizer.get_default_param_dict()

    # add def synthetizer params
    for k, v in default_params.items():
        parser.add_argument('--{}'.format(k), type=float, default=v)

    parser.add_argument('-n', '--num_items', type=int, default=1000)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-s', '--subset', type=str, default='train')
    parser.add_argument('tips_path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_command_line()

    syn = Synthesizer(**vars(args))
    entries = generate_dataset(syn, num_items=args.num_items, num_workers=args.num_workers)
    save_dataset(syn, entries, args)

    # for num_workers in range(0, 17, 4):
    # for num_workers in range(0, 17, 4):
    #     start_time = time.time()
    #     syn = Synthesizer(tip_pkl_path, num_workers=num_workers)
    #     generate_dataset(syn, 3000, num_workers)
    #     print("For {} workers comp took {}".format(num_workers, time.time() - start_time))
