import cv2
import numpy as np

from synthetizer.shapes import create_bubble_array
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
        y = np.clip(int(height / 2 + height * 0.01 * np.random.randn()), 0, height - 1)
        x = np.clip(int(width / 2 + width * 0.01 * np.random.randn()), 0, width - 1)

        # y = np.random.binomial(height, 0.5)
        # x = np.random.binomial(width, 0.5)
        ifft_canvas[y, x] = np.random.randn() + np.random.randn() * 1.0j

    canvas = normalize(np.real(np.fft.ifft2(ifft_canvas)))

    return canvas


def apply_linear_f(x, min_t, max_t):
    x -= min_t
    x /= max_t - min_t
    x = np.clip(x, 0.0, 1.0)
    return x


if __name__ == '__main__':
    for i in range(10):
        # noise = generate_value_noise(128, 128, np.random.randint(3, 10), np.random.randint(3, 10), np.random.randint(2, 8))
        # noise = generate_value_noise(256, 256, np.random.randint(1, 9), np.random.randint(1, 10), np.random.randint(1, 10))
        # noise = 1.0 * (noise > 0.5)
        noise_periodic = generate_periodic_noise(256, 256, 5)
        noise_value = generate_value_noise(256, 256, np.random.randint(1, 9), np.random.randint(1, 10), np.random.randint(1, 10))

        # noise = generate_periodic_noise(512, 512, np.random.randint(1, 8))
        cv2.imshow("noise", apply_linear_f(0.8 * noise_periodic + 0.2 * noise_value, 0.3, 0.9))
        cv2.waitKey(0)

    # noise = generate_random_value_noise(512, 512, 4, 5, 3)
    # noise = generate_random_value_noise(512, 512, 4, 5, 3)
