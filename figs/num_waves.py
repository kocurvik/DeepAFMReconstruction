import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from synth.generator import apply_linear_f
from utils.image import normalize


def generate_single_canvas_periodic_noise(num_waves):
    ifft_canvas = 1.0j * np.zeros([512, 512])

    periodic_binomial_p = 0.02

    for _ in range(num_waves):
        y = np.random.binomial(512 // 2, periodic_binomial_p)
        x = np.random.binomial(512 // 2, periodic_binomial_p)
        a = np.random.randn()
        b = np.random.randn()
        ifft_canvas[y, x] = a + b * 1.0j

    canvas = normalize(np.abs(np.fft.ifft2(ifft_canvas)))

    return canvas


def generate():
    num_noises = min(np.random.zipf(self.num_noises_zipf_a), 20)
    noises = np.empty([num_noises, 512, 512])

    t_beta_low = 2 
    t_beta_high = 4

    for i in range(num_noises):
        num_waves = min(np.random.zipf(1.5) + 1, 100)
        noise = generate_single_canvas_periodic_noise(num_waves)
        min_t = np.random.beta(t_beta_low, t_beta_high)
        max_t = np.random.beta(t_beta_high, t_beta_low)
        # max_t = np.random.rand()
        noise = apply_linear_f(noise, min_t, max_t)
        noises[i] = noise

    noise_weights = np.random.dirichlet(np.ones(num_noises))
    noise = np.average(noises, weights=noise_weights, axis=0)

    noise = ndimage.rotate(noise, np.random.rand() * 90, reshape=False)

    x_crop = np.random.randint(512 - 3 * 128, 512 - 2 * 128)
    y_crop = np.random.randint(512 - 3 * 128, 512 - 2 * 128)

    return noise[y_crop: y_crop + 128, x_crop: x_crop + 128]


def save_fig(image, filename):
    plt.plot(image[256, :])
    plt.ylim([0, 1])
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.cla()


if __name__ == '__main__':
    # num_waves_list = [1, 2, 5, 10, 20]
    # for num_waves in num_waves_list:
    #     image = generate_single_canvas_periodic_noise(num_waves + 1)
    #     cv2.imwrite('map_{}.png'.format(num_waves), (255 * image).astype(np.uint8))
    #     save_fig(image, 'map_{}.pdf'.format(num_waves))
    # cv2.waitKey(0)

    while True:
        num_noises = 2
        noises = np.empty([num_noises, 512, 512])

        t_beta_low = 2
        t_beta_high = 4

        for i in range(num_noises):
            num_waves = min(np.random.zipf(1.5) + 1, 100)
            noise = generate_single_canvas_periodic_noise(num_waves)

            cv2.imwrite('smooth_{}.png'.format(i), (255 * noise).astype(np.uint8))
            save_fig(noise, 'smooth_{}.pdf'.format(i))

            min_t = np.random.beta(t_beta_low, t_beta_high)
            max_t = np.random.beta(t_beta_high, t_beta_low)
            # max_t = np.random.rand()
            noise = apply_linear_f(noise, min_t, max_t)

            cv2.imwrite('thresholded_{}.png'.format(i), (255 * noise).astype(np.uint8))
            save_fig(noise, 'thresholded_{}.pdf'.format(i))

            noises[i] = noise

        noise_weights = np.random.dirichlet(np.ones(num_noises))
        noise = normalize(np.average(noises, weights=noise_weights, axis=0))

        cv2.imwrite('final.png', (255 * noise).astype(np.uint8))
        save_fig(noise, 'final.pdf')

        cv2.imshow('noise', noise)
        cv2.waitKey(0)