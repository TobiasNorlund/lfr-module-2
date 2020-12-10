import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
import cv2
from .image_io import crop_patch

def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win


class MOSSETracker:

    def __init__(self, std=20, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.width = None
        self.height = None
        self.std = std
        self.region = None
        self.A = None
        self.B = None

    @staticmethod
    def normalize(image):
        normalized_image = np.log(1+image)
        normalized_image = (normalized_image - np.mean(normalized_image)) / (np.std(normalized_image) + 1e-5)
        window = window_func_2d(image.shape[0], image.shape[1])
        normalized_image = normalized_image * window
        return normalized_image

    @staticmethod
    def get_fourier_transformed_gaussian(height, width, std, mean_x, mean_y):
        v, u = np.meshgrid(range(width), range(height))
        gaussian_ft = np.exp( \
            -2 * np.pi**2 * std**2 * (u**2 + v**2) / (width*height) \
            -2j*np.pi*(mean_y*u/height + mean_x*v/width))
        return gaussian_ft

    @staticmethod
    def get_fft2_gaussian(height, width, std, mean_x, mean_y):
        xy = np.mgrid[0:height, 0:width].reshape(2,-1).T
        gaussian = multivariate_normal(mean=np.array([mean_y, mean_x]), cov=np.eye(2)*std**2).pdf(xy)
        gaussian = np.array(gaussian).reshape(height, width)
        return fft2(gaussian)

    def start(self, image, region):
        """
        Construct initial model (=filter) in fourier domain using provided region in image
        """
        assert len(image.shape) == 2, "Only grayscale images supported atm"

        patch = crop_patch(image, region)

        # Where the gaussian should be centered
        self.region = region
        mean_x = self.region.width // 2
        mean_y = self.region.height // 2

        C = MOSSETracker.get_fft2_gaussian(height=region.height, width=region.width, std=self.std,
                                           mean_x=mean_x, mean_y=mean_y)
        P = fft2(MOSSETracker.normalize(patch))

        self.A = np.conjugate(C) * P
        self.B = np.conjugate(P) * P
        self.M = self.A / self.B

    def detect(self, image):
        """
        Find the object's new position in image, using current model M
        """
        assert len(image.shape) == 2, "Only grayscale images supported atm"

        patch = crop_patch(image, self.region)

        P = fft2(MOSSETracker.normalize(patch))
        response = ifft2( np.conjugate(self.M) * P )
        # response should be a slightly shifted gaussian
        y, x = np.unravel_index(np.argmax(response), response.shape)

        self.region.xpos += x - self.region.width // 2
        self.region.ypos += y - self.region.height // 2

        return response

    def update(self, image):
        """
        Re-fit model M using new object position found in self.region (from detection step)
        """

        patch = crop_patch(image, self.region)

        mean_x = self.region.width // 2
        mean_y = self.region.height // 2

        C = MOSSETracker.get_fft2_gaussian(height=self.region.height, width=self.region.width, std=self.std,
                                           mean_x=mean_x, mean_y=mean_y)
        P = fft2(MOSSETracker.normalize(patch))

        self.A = self.A * (1-self.learning_rate) + np.conjugate(C) * P * self.learning_rate
        self.B = self.B * (1-self.learning_rate) + np.conjugate(P) * P * self.learning_rate
        self.M = self.A / self.B


class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    def start(self, image, region):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.crop_patch(image)

        patch = patch/255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)

        self.template = fft2(patch)

    def detect(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)

        responsef = np.conj(self.template) * patchf
        response = ifft2(responsef)

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image, lr=0.1):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr
