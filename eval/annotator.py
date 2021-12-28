import os
import pickle

import cv2
import numpy as np

from utils.image import load_lr_img_from_gwy, normalize


def annotate_entries(entries):
    for i, entry in enumerate(entries):

        print("Entry filename: ", entry['filename'])

        if i > 0:
            cv2.imshow("Chosen points!", example_img)

        clicked = []
        def mouse_callback(event, x, y, flags, params):
            if event == 1:
                clicked.append([x, y])
                print("Clicked x: {}, y: {}".format(x, y))

        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('image', mouse_callback)
        cv2.imshow('image', normalize(entry['img_l']))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        entry['registration_points'] = clicked

        if i == 0:
            example_img = cv2.cvtColor(normalize(entry['img_l']), cv2.COLOR_GRAY2BGR)
            for pt in clicked:
                example_img = cv2.circle(example_img, (pt[0], pt[1]), 3, (0, 0, 255), thickness=2)

    return entries


def extract_eval_data(path):
    filenames = os.listdir(path)
    filenames = [f for f in filenames if '.gwy' in f and 'tip' not in f and 'bad' not in f]

    entries = []

    for filename in filenames:
        gwy_path = os.path.join(path, filename)
        img_l, img_r = load_lr_img_from_gwy(gwy_path, normalize_range=False)

        entry_dict = {'filename': filename, 'gwy_path': gwy_path, 'img_l': img_l, 'img_r': img_r}

        entries.append(entry_dict)
    return entries


def save_entries(entries, path):
    pkl_path = os.path.join(path, 'entries.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(entries, f)


if __name__ == '__main__':
    # path = 'D:/Research/data/GEFSEM/2021-04-07 - Dataset/Neno'
    path = 'D:/Research/data/GEFSEM/EvalData/TGZ3'
    entries = extract_eval_data(path)
    annotate_entries(entries)
    save_entries(entries, path)
