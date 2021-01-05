import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from cvl.dataset import BoundingBox
import cv2
from cvl.image_io import crop_patch


def add_cli_arguments(parser):
    group = parser.add_argument_group("MOSSE", "MOSSE Tracker arguments")
    group.add_argument("--learning-rate", type=float, default=0.1)
    group.add_argument("--std", type=float, default=1.0)


def get_tracker(args):
    return MOSSETrackerGrayscale(learning_rate=args.learning_rate, std=args.std)


def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win


class MOSSETrackerGrayscale:
    def __init__(self, std=1.0, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.width = None
        self.height = None
        self.std = std
        self.region = None
        self.region_center = None
        self.A = None
        self.B = None

    @staticmethod
    def normalize(patch):
        #patch = patch + np.ones(patch.shape)
        patch = np.log(patch+1)
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        window = window_func_2d(patch.shape[0], patch.shape[1])
        patch = patch * window
        return patch

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
        # Convert to grayscale
        if len(image.shape)==3: image = np.sum(image, 2) / 3

        # Where the gaussian should be centered
        self.region = region
        self.region_center = (region.height // 2, region.width // 2)

        C = MOSSETrackerGrayscale.get_fourier_transformed_gaussian(height=region.height, width=region.width, std=self.std,
                                           mean_x=0, mean_y=0)
        patch = crop_patch(image, region)

        P = fft2(MOSSETrackerGrayscale.normalize(patch))
        self.A = np.conjugate(C) * P
        self.B = np.conjugate(P) * P

        self.M = self.A / self.B

    def detect(self, image):
        """
        Find the object's new position in image, using current model M
        """
        # Convert to grayscale
        if len(image.shape)==3: image = np.sum(image, 2) / 3

        self.compute_response(image)
        r, c = np.unravel_index(np.argmax(self.last_response), self.last_response.shape)

        # Keep for visualisation

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        # Revert update if bbox is completely out of image
        if self.region.intersection_box(BoundingBox("tl-size", 0, 0, image.shape[1], image.shape[0])).area() == 0.0:
            self.region.ypos -= r_offset
            self.region.xpos -= c_offset

        return self.region

    def compute_response(self, image): #Now redunant
        """
            Update docstring
        """
        patch = crop_patch(image, self.region)

        P = fft2(MOSSETrackerGrayscale.normalize(patch))
        response = ifft2( np.conjugate(self.M) * P )

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response
        return self

    def update(self, image, B=None):
        """
        Re-fit model M using new object position found in self.region (from detection step)
        """
        # Convert to grayscale
        if len(image.shape)==3: image = np.sum(image, 2) / 3
        patch = crop_patch(image, self.region)
        normalized_patch = MOSSETrackerGrayscale.normalize(patch)

        C = MOSSETrackerGrayscale.get_fourier_transformed_gaussian(height=self.region.height, width=self.region.width, std=self.std,
                                           mean_x=0, mean_y=0)
        P = fft2(normalized_patch)

        self.A = self.A * (1-self.learning_rate) + np.conjugate(C) * P * self.learning_rate

        if B is None:
            self.B = self.B * (1-self.learning_rate) + np.conjugate(P) * P * self.learning_rate
        else:
            self.B = B

        self.M = self.A / self.B

        return normalized_patch

    def get_filter(self, image):
        region = self.region
 
        if len(image.shape)==3 : image = np.sum(image, axis=2) / 3

        patch = crop_patch(image, region)
        window = patch
        filt = np.abs(ifft2(self.M))
        try:
            response = np.abs(self.last_response)
        except:
            response = np.zeros(filt.shape)
        return window, filt, response