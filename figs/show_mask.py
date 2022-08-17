import pickle

import cv2
import numpy as np

from utils.image import normalize_joint, normalize


def main():
    entries_path = 'D:/Research/data/GEFSEM/UltraMicroscopy2022Q2/EvalMasked/Neno/entries.pkl'
    print("Loading data from path: ", entries_path)
    data_basename = dir
    with open(entries_path, 'rb') as f:
        entries = pickle.load(f)

    entry = entries[2]

    img_l, img_r = normalize_joint([entry['img_l'], entry['img_r']])

    img_l = (255 * img_l).astype(np.uint8)
    img_r = (255 * img_r).astype(np.uint8)

    mask = (255 * normalize(entry['mask'])).astype(np.uint8)

    masked_l = np.repeat(img_l[:, :, np.newaxis], 3, axis=-1)
    masked_r = np.repeat(img_r[:, :, np.newaxis], 3, axis=-1)

    masked_l[:, :, 1] = np.where(mask > 0, 255, masked_l[:, :, 1])
    masked_r[:, :, 1] = np.where(mask > 0, 255, masked_r[:, :, 1])

    cv2.imwrite('figs/show_mask/img_l.png', img_l)
    cv2.imwrite('figs/show_mask/img_r.png', img_r)
    cv2.imwrite('figs/show_mask/mask.png', mask)
    cv2.imwrite('figs/show_mask/masked_l.png', masked_l)
    cv2.imwrite('figs/show_mask/masked_r.png', masked_r)

if __name__ == '__main__':
    main()