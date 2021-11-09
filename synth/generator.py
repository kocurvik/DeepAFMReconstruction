import cv2
import numpy as np
from matplotlib import pyplot as plt


def generate_grid_structure(width=128, height=128):
    canvas = np.zeros([4 * height, 4 * width], dtype=np.float32)

    y, x = np.mgrid[:4 * height, :4 * width].astype(np.float32)

    # shift randomly
    y -= np.random.uniform(0, 4*height)
    x -= np.random.uniform(0, 4*width)

    num_lines = np.random.randint(1, 3)

    additive = True if np.random.rand() > 0.5 else False

    for _ in range(num_lines):
        line_height = np.random.rand()

        # line_width = np.random.randint(1, height)
        # line_space = np.random.randint(1, width)
        line_width = np.random.triangular(0, height * 2, 4 * height)
        line_space = np.random.triangular(0, height * 2, 4 * height)

        angle = np.random.rand() * 2 * np.pi
        cos = np.cos(angle)
        sin = np.sin(angle)

        if additive:
            canvas = np.where((cos * y + sin * x) % (line_width + line_space) < line_width, canvas + line_height, canvas)
        else:
            canvas = np.where((cos * y + sin * x) % (line_width + line_space) < line_width, line_height, canvas)

    canvas /= np.max(canvas)
    canvas = cv2.resize(canvas, (width, height), cv2.INTER_AREA)

    # y_min = np.random.randint(0, 3 * height)
    # x_min = np.random.randint(0, 3 * width)

    return canvas

if __name__ == '__main__':
    for i in range(100):
        canvas = generate_grid_structure(128, 128)
        cv2.imshow("Canvas", canvas / np.max(canvas))
        cv2.waitKey(0)