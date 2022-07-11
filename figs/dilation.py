from scipy.ndimage.morphology import grey_dilation
from matplotlib import pyplot as plt

import numpy as np

from synth.tip_dilation import dilate


def generate_dilation_graphs():
    y = np.zeros(1000)

    x = np.arange(1000)

    y[200:400] = 0.5
    y[600:800] = -0.5

    tip_thin = - np.linspace(-1, 1, 100) ** 2
    tip_wide = - np.linspace(-1, 1, 250) ** 2

    dilated_thin = dilate(y[np.newaxis, :], tip_thin[np.newaxis, :])
    dilated_wide = dilate(y[np.newaxis, :], tip_wide[np.newaxis, :])
    plt.plot(x, dilated_thin[0], 'r')
    plt.plot(x, dilated_wide[0], 'b')
    plt.plot(x, y, 'k')
    plt.plot(x[:100], -tip_thin)
    plt.plot(x[:250], -tip_wide)
    plt.show()



if __name__ == '__main__':
    generate_dilation_graphs()
