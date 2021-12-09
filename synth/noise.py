import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from utils.image import normalize


def generate_value_noise(width, height, init_width, init_height, octaves):
    canvas = np.zeros([height, width])

    for i in range(1, octaves + 1):
        seed = np.random.rand(init_width * i, init_height * i) / octaves
        canvas += cv2.resize(seed, (width, height), interpolation=cv2.INTER_CUBIC)

    return canvas


def generate_periodic_noise(height, width, num_waves):
    ifft_canvas = 1.0j * np.zeros([height, width])

    for _ in range(num_waves):
        # y = np.clip(int(height / 2 + height * 0.05 * np.random.randn()), 0, height - 1)
        # x = np.clip(int(width / 2 + width * 0.05 * np.random.randn()), 0, width - 1)

        y = np.random.binomial(height // 2, 0.02)
        x = np.random.binomial(width // 2, 0.02)
        # y = int(np.floor(height * np.random.beta(1, 100) - 1e-8))
        # print(y)
        # x = int(np.floor(width * np.random.beta(1, 100) - 1e-8))
        a = np.random.randn()
        b = np.random.randn()
        ifft_canvas[y, x] = a + b * 1.0j

    # fft_canvas = np.abs((np.fft.ifft2(ifft_canvas)))
    canvas = normalize(np.abs(np.fft.ifft2(ifft_canvas)))

    return canvas


def apply_linear_f(x, min_t, max_t):
    if min_t > max_t:
        return np.where(x < min_t, 0.0, 1.0)
    else:
        x -= min_t
        x /= max_t - min_t
        x = np.clip(x, 0.0, 1.0)
        return x


if __name__ == '__main__':
    for i in range(100):
        # noise = generate_value_noise(128, 128, np.random.randint(3, 10), np.random.randint(3, 10), np.random.randint(2, 8))
        # noise = generate_value_noise(256, 256, np.random.randint(1, 9), np.random.randint(1, 10), np.random.randint(1, 10))
        # noise = 1.0 * (noise > 0.5)
        # noise_periodic = generate_periodic_noise(256, 256, np.random.randint(2, 5))

        resolution = 128

        gen_resolution = 4 * resolution

        num_noises = min(np.random.zipf(2), 20)
        noises = np.empty([num_noises, gen_resolution, gen_resolution])

        for i in range(num_noises):
            num_waves = min(np.random.zipf(1.5) + 1, 100)
            noise = generate_periodic_noise(gen_resolution, gen_resolution, num_waves)
            min_t = np.random.beta(2, 4)
            max_t = np.random.beta(4, 2)
            # max_t = np.random.rand()
            noise = apply_linear_f(noise, min_t, max_t)
            noises[i] = noise

            # print("Noise: {}, min t: {}, max t: {}, num waves")

            # cv2.imshow("noise {}".format(i), noises[i])

        noise_weights = np.random.dirichlet(np.ones(num_noises))
        noise = np.average(noises, weights=noise_weights, axis=0)

        # noise = generate_periodic_noise(512, 512, np.random.randint(1, 8))
        # cv2.imshow("noise", apply_linear_f(0.8 * noise_periodic + 0.2 * noise_value, 0.3, 0.9))

        noise = ndimage.rotate(noise, np.random.rand()*90, reshape=False)

        cv2.imshow("noise orig", noise)
        x_crop = np.random.randint(gen_resolution - 3 * resolution,  gen_resolution - 2 * resolution)
        y_crop = np.random.randint(gen_resolution - 3 * resolution,  gen_resolution - 2 * resolution)
        cv2.imshow("noise small", noise[y_crop: y_crop + resolution, x_crop: x_crop + resolution])
        cv2.waitKey(0)

        # plt.plot(noise[100, :])
        # plt.show()


    # noise = generate_random_value_noise(512, 512, 4, 5, 3)
    # noise = generate_random_value_noise(512, 512, 4, 5, 3)
