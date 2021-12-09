import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from utils.image import normalize


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


class GridGenerator:
    def __init__(self, **kwargs):
        for (prop, default) in GridGenerator.get_default_param_dict().items():
            setattr(self, prop, kwargs.get(prop, default))

    @staticmethod
    def get_default_param_dict():
        default_params = {'resolution': 128}
        return default_params

    def generate(self):
        return generate_grid_structure(self.resolution, self.resolution)

# if __name__ == '__main__':
#     for i in range(100):
#         canvas = generate_grid_structure(128, 128)
#         cv2.imshow("Canvas", canvas / np.max(canvas))
#         cv2.waitKey(0)


def apply_linear_f(x, min_t, max_t):
    if min_t > max_t:
        return np.where(x < min_t, 0.0, 1.0)
    else:
        x -= min_t
        x /= max_t - min_t
        x = np.clip(x, 0.0, 1.0)
        return x

class FFTGenerator:
    def __init__(self, **kwargs):
        for (prop, default) in FFTGenerator.get_default_param_dict().items():
            setattr(self, prop, kwargs.get(prop, default))

        self.gen_resolution = 4 * self.resolution

    @staticmethod
    def get_default_param_dict():
        default_params = {'resolution': 128,
                          'num_noises_zipf_a': 2, 'num_waves_zipf_a': 1.5,
                          't_beta_low': 2, 't_beta_high': 4,
                          'periodic_binomial_p': 0.02}
        return default_params

    def generate_single_canvas_periodic_noise(self, num_waves):
        ifft_canvas = 1.0j * np.zeros([self.gen_resolution, self.gen_resolution])

        for _ in range(num_waves):
            y = np.random.binomial(self.gen_resolution // 2, self.periodic_binomial_p)
            x = np.random.binomial(self.gen_resolution // 2, self.periodic_binomial_p)
            a = np.random.randn()
            b = np.random.randn()
            ifft_canvas[y, x] = a + b * 1.0j

        canvas = normalize(np.abs(np.fft.ifft2(ifft_canvas)))

        return canvas


    def generate(self):
        num_noises = min(np.random.zipf(self.num_noises_zipf_a), 20)
        noises = np.empty([num_noises, self.gen_resolution, self.gen_resolution])

        for i in range(num_noises):
            num_waves = min(np.random.zipf(self.num_waves_zipf_a) + 1, 100)
            noise = self.generate_single_canvas_periodic_noise(num_waves)
            min_t = np.random.beta(self.t_beta_low, self.t_beta_high)
            max_t = np.random.beta(self.t_beta_high, self.t_beta_high)
            # max_t = np.random.rand()
            noise = apply_linear_f(noise, min_t, max_t)
            noises[i] = noise

        noise_weights = np.random.dirichlet(np.ones(num_noises))
        noise = np.average(noises, weights=noise_weights, axis=0)

        noise = ndimage.rotate(noise, np.random.rand() * 90, reshape=False)

        x_crop = np.random.randint(self.gen_resolution - 3 * self.resolution, self.gen_resolution - 2 * self.resolution)
        y_crop = np.random.randint(self.gen_resolution - 3 * self.resolution, self.gen_resolution - 2 * self.resolution)

        return noise[y_crop: y_crop + self.resolution, x_crop: x_crop + self.resolution]
