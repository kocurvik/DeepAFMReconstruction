import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.image import load_lr_img_from_gwy, normalize_joint

if __name__ == '__main__':
    img_kremik_l, img_kremik_r = load_lr_img_from_gwy('D:/Research/data/GEFSEM/EvalData/Kremik/5x5_t-d_0deg_210804_152052.gwy', normalize_range=True)

    entries_path = 'D:/Research/data/GEFSEM/EvalData/Kremik/entries.pkl'
    print("Loading data from path: ", entries_path)
    with open(entries_path, 'rb') as f:
        entries = pickle.load(f)

    row = 0
    i = 0

    while True:

        img_kremik_l = entries[i]['img_l']
        img_kremik_r = entries[i]['img_r']

        img_vis_l, img_vis_r = normalize_joint([img_kremik_l, img_kremik_r])

        img_line_l = np.copy(img_vis_l)
        img_line_l[row, :] = 1.0

        cv2.imshow("L", img_vis_l)
        cv2.imshow("Line", img_line_l)
        cv2.imshow("R", img_vis_r)
        key = cv2.waitKey(0)

        if key == ord('u'):
            row = max(0, row - 10)

        if key == ord('d'):
            row = min(511, row + 10)

        if key == ord('n'):
            i+=1

        if key == ord('s'):

            cv2.imwrite("figs/shadows_actual/l.png", 255 * img_vis_l)
            cv2.imwrite("figs/shadows_actual/r.png", 255 * img_vis_r)

            plt.plot(img_vis_l[row, :])
            plt.plot(img_vis_r[row, :])
            plt.ylim([0, 1])
            plt.margins(x=0)
            plt.savefig("figs/shadows_actual/plot_l.pdf", bbox_inches='tight')
            # plt.show()
            plt.cla()