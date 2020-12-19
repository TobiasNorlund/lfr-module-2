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
        self.A = None
        self.B = None

    @staticmethod
    def normalize(patch):
        patch = np.log(1+patch)
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
    def gen_affine_transform(image, n_transforms=15, rot_sd=0.2):
        transformed_images = []
        for i in range(n_transforms):
            alpha = np.random.normal(loc=0, scale=rot_sd)
            M = np.float32([
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0]])
            transformed_images.append(cv2.warpAffine(image, M, image.shape))
        return transformed_images

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
        image = np.sum(image, 2) / 3

        # Where the gaussian should be centered
        self.region = region
        self.region_center = (region.height // 2, region.width // 2)
        mean_x = 0 #self.region.width // 2
        mean_y = 0 #self.region.height // 2

        C = MOSSETrackerGrayscale.get_fourier_transformed_gaussian(height=region.height, width=region.width, std=self.std,
                                           mean_x=mean_x, mean_y=mean_y)

        patch = crop_patch(image, region)

        P = fft2(MOSSETrackerGrayscale.normalize(patch))
        self.A = np.conjugate(C) * P
        self.B = np.conjugate(P) * P

        #aff_images = MOSSETrackerGrayscale.gen_affine_transform(patch, n_transforms=0)
        #for aff_patch in aff_images:
        #    P = fft2(MOSSETrackerGrayscale.normalize(patch))
        #    self.A += C * np.conjugate(P)
        #    self.B += np.conjugate(P) * P

        self.M = self.A / self.B

    def detect(self, image):
        """
        Find the object's new position in image, using current model M
        """
        # Convert to grayscale
        image = np.sum(image, 2) / 3

        patch = crop_patch(image, self.region)

        P = fft2(MOSSETrackerGrayscale.normalize(patch))
        response = ifft2( np.conjugate(self.M) * P )
        # response should be a slightly shifted gaussian
        #y, x = np.unravel_index(np.argmax(response), response.shape)

        #delta_x = x - self.region.width // 2
        #delta_y = y - self.region.height // 2
        #print(f"delta_x: {delta_x}\tdelta_y:{delta_y}")

        #self.region.xpos += delta_x
        #self.region.ypos += delta_y

        # ---
        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset
        #print(f"r_offset: {r_offset}\tc_offset: {c_offset}")
        # ---

        # Revert update if bbox is completely out of image
        if self.region.intersection_box(BoundingBox("tl-size", 0, 0, image.shape[1], image.shape[0])).area() == 0.0:
            self.region.ypos -= r_offset
            self.region.xpos -= c_offset

        return self.region

    def update(self, image):
        """
        Re-fit model M using new object position found in self.region (from detection step)
        """
        # Convert to grayscale
        image = np.sum(image, 2) / 3
        patch = crop_patch(image, self.region)
        normalized_patch = MOSSETrackerGrayscale.normalize(patch)

        mean_x = 0 # self.region.width // 2
        mean_y = 0 # self.region.height // 2

        C = MOSSETrackerGrayscale.get_fourier_transformed_gaussian(height=self.region.height, width=self.region.width, std=self.std,
                                           mean_x=mean_x, mean_y=mean_y)
        P = fft2(normalized_patch)

        self.A = self.A * (1-self.learning_rate) + np.conjugate(C) * P * self.learning_rate
        self.B = self.B * (1-self.learning_rate) + np.conjugate(P) * P * self.learning_rate
        self.M = self.A / self.B

        return normalized_patch

