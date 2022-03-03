# !!! Do not import torch here or in any subimpots!
# This commits too much memory even without using and mp version then runs out of RAM very quickly!
# For details see https://stackoverflow.com/questions/64837376/how-to-efficiently-run-multiple-pytorch-processes-models-at-once-traceback
import argparse
import json
import os
from datetime import datetime
from hashlib import sha1
from multiprocessing import Pool

import git
import numpy as np

from synth.artifacts import Artifactor
from synth.generator import GridGenerator, FFTGenerator
from synth.tip_dilation import FastTipDilator
from utils.image import normalize_joint, subtract_mean_plane_both, subtract_mean_plane


class Synthesizer():
    """
    Synthesizer class. For params description, see default params dict in the end of source file
    """

    def __init__(self, tips_path=None, **kwargs):
        self.entries = []

        default_params = self.get_default_param_dict()
        # for hashing
        self.param_names = sorted(list(default_params.keys()))

        for (prop, default) in default_params.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.generator = FFTGenerator(**self.get_param_dict())
        self.tip_dilator = FastTipDilator(tips_path, **self.get_param_dict())
        self.artifactor = Artifactor(**self.get_param_dict())

    @staticmethod
    def get_default_param_dict():
        default_params = {'subtract_mean_plane': 1.0}

        default_params.update(FFTGenerator.get_default_param_dict())
        default_params.update(FastTipDilator.get_default_param_dict())
        default_params.update(Artifactor.get_default_param_dict())

        return default_params

    @staticmethod
    def get_param_names():
        return sorted(list(Synthesizer.get_default_param_dict().keys()))

    def get_param_dict(self):
        return {param_name: getattr(self, param_name) for param_name in self.param_names}

    def get_param_hash(self):
        dict = self.get_param_dict()
        json_string = json.dumps(dict, sort_keys=True)
        return sha1(json_string.encode('utf-8')).hexdigest()

    def generate_entries(self, n):
        for i in range(n):
            entry = self.generate_single_entry(apply_artifacts=False)
            self.entries.append(entry)
        return np.array(self.entries)

    def generate_entries_w_artifacts(self, n):
        for i in range(n):
            entry = self.generate_single_entry(apply_artifacts=True)
            self.entries.append(entry)
        return np.array(self.entries)

    def generate_single_entry(self, apply_artifacts=False):
        image_gt = self.generator.generate()
        image_dil = self.tip_dilator.apply(image_gt)

        if apply_artifacts:
            image_l, image_r = self.artifactor.apply(image_dil)
            if self.subtract_mean_plane:
                image_l, image_r, plane = subtract_mean_plane_both(image_l, image_r, return_plane=True)
                image_gt -= plane
            images = normalize_joint([image_l, image_r, image_gt])
        else:
            if self.subtract_mean_plane:
                image_dil, plane = subtract_mean_plane(image_dil,return_plane=True)
                image_gt -= plane
            images = normalize_joint([image_dil, image_gt])

        entry = np.stack(images, axis=0).astype(np.float32)
        return entry


def generate_dataset(synthetizer, num_items, num_workers=8, apply_artifacts=False):
    if num_workers > 1:
        pool = Pool(num_workers)
        num_per_worker = [num_items // num_workers + (num_items % num_workers > i) for i in range(num_workers)]
        print("Workers will generate ", num_per_worker, "items")
        if apply_artifacts:
            entries = pool.map(synthetizer.generate_entries_w_artifacts, num_per_worker)
        else:
            entries = pool.map(synthetizer.generate_entries, num_per_worker)
        entries = np.concatenate(entries, axis=0)
        return entries
    else:
        print("Single worker will generete {} items".format(num_items))
        if apply_artifacts:
            entries = synthetizer.generate_entries_w_artifacts(num_items)
        else:
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

    parser.add_argument('-n', '--num_items', type=int, default=1000, help='Number of items to generate')
    parser.add_argument('-nw', '--num_workers', type=int, default=1, help='Number of workers to be used in multiprocessing')
    parser.add_argument('-s', '--subset', type=str, default='train', help='Subset - train/val/test')
    parser.add_argument('-aa', '--apply_artifacts', action='store_true', default=False, help='Whether to apply artifacts, if they are not applied this allows for them to be applied on the fly')
    parser.add_argument('tips_path', type=str, help='Path to the pickle of the tips')
    parser.add_argument('out_path', type=str, help='Path where a new folder containing the dataset will be generated')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # This generates a dataset json and npy file. In the final version only
    # The parser has multiple parameters which can be seen in the Artifactor, FastTipDilator and FFTGenerator objects.
    # Example usage: python synth/synthetizer.py -n 5000 -nw 8 -s val -aa --shadows_prob 1.0 --overshoot_prob 0.0 --noise_prob 0.5 /path/to/tips.pkl /path/where/dataset/will/be/generated/
    args = parse_command_line()
    syn = Synthesizer(**vars(args))
    entries = generate_dataset(syn, num_items=args.num_items, num_workers=args.num_workers, apply_artifacts=args.apply_artifacts)
    save_dataset(syn, entries, args)
