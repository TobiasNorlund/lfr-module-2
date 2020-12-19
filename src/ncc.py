import numpy as np
from cvl.dataset import BoundingBox
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import cv2
from cvl.image_io import crop_patch


def add_cli_arguments(parser):
    group = parser.add_argument_group("NCC", "NCC Tracker arguments")
    group.add_argument("--learning-rate", type=float, default=0.1)


def get_tracker(args):
    return NCCTracker(learning_rate=args.learning_rate)


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
        # Convert to grayscale
        image = np.sum(image, 2) / 3
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.crop_patch(image)

        patch = patch/255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)

        self.template = fft2(patch)

    def detect(self, image):
        # Convert to grayscale
        image = np.sum(image, 2) / 3
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

        # Revert update if bbox is completely out of image
        if self.region.intersection_box(BoundingBox("tl-size", 0, 0, image.shape[1], image.shape[0])).area() == 0.0:
            self.region.ypos -= r_offset
            self.region.xpos -= c_offset

        return self.region

    def update(self, image, lr=0.1):
        # Convert to grayscale
        image = np.sum(image, 2) / 3
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr
