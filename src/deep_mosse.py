import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
import torch
from PIL import Image
from torchvision import transforms
from cvl.dataset import BoundingBox
import cv2

from cvl.image_io import crop_patch
from cvl.features import alexnetFeatures


def add_cli_arguments(parser):
    group = parser.add_argument_group("DeepMOSSE", "DeepMOSSE Tracker arguments")
    group.add_argument("--learning-rate", type=float, default=0.5)
    group.add_argument("--std", type=float, default=2.0)
    group.add_argument("--layer-num", type=int, default=1)


def get_tracker(args):
    return MOSSETrackerDeep(learning_rate=args.learning_rate, std=args.std, layer_num=args.layer_num)


def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win


class MOSSETrackerDeep:
    def __init__(self, std=1.0, learning_rate=0.15, layer_num=1):  # RELU layers = 1, 4, 7, 9, 11
        self.learning_rate = learning_rate
        self.width = None
        self.height = None
        self.std = std
        self.region = None
        self.region_center = None
        self.A = None
        self.B = None
        self.C = None
        self.preprocess = MOSSETrackerDeep.preprocess()

        # Load AlexNet
        self.alexnet = alexnetFeatures(pretrained=True)

        # Set the hooks for activations
        self.activation = {}
        self.alexnet.features[layer_num].register_forward_hook(self.get_activation('conv'))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def cfft2(self, patch):
        patch_shape = patch.shape[1::-1]
        preprocessed_patch = self.preprocess(Image.fromarray(patch))
        _ = self.alexnet(preprocessed_patch.unsqueeze(0))
        acts = self.activation['conv'].numpy()[0]
        norm_patches = [MOSSETrackerDeep.normalize(acts[idx]) for idx in range(acts.shape[0])]
        norm_patches = [Image.fromarray(patch).resize(patch_shape) for patch in norm_patches]
        norm_patches = [np.array(patch) for patch in norm_patches]  # TODO
        return fft2(norm_patches, axes=(-2, -1))

    @staticmethod
    def preprocess():
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess

    @staticmethod
    def normalize(patch):
        # patch = np.log(1+patch)
        # patch = patch - np.mean(patch)
        # patch = patch / np.std(patch)
        window = window_func_2d(patch.shape[0], patch.shape[1])
        patch *= window
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
        # Where the gaussian should be centered
        self.region = region
        self.region_center = (region.height // 2, region.width // 2)

        self.C = MOSSETrackerDeep.get_fourier_transformed_gaussian(height=region.height, width=region.width, std=self.std,
                                           mean_x=0, mean_y=0)

        patch = crop_patch(image, region)

        Ps = self.cfft2(patch)  #TODO
        self.A = np.mean([np.conjugate(self.C) * P for P in Ps], axis=0)
        self.B = np.mean([np.conjugate(P) * P for P in Ps], axis=0)

        self.M = self.A / self.B

    def detect(self, image):
        """
        Find the object's new position in image, using current model M
        """
        patch = crop_patch(image, self.region)

        Ps = self.cfft2(patch)  #TODO

        response = np.mean([ifft2(np.conjugate(self.M) * P) for P in Ps], axis=0)

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

    def update(self, image):
        """
        Re-fit model M using new object position found in self.region (from detection step)
        """
        patch = crop_patch(image, self.region)

        # C = MOSSETrackerDeep.get_fourier_transformed_gaussian(height=self.region.height, width=self.region.width, std=self.std,
        #                                    mean_x=0, mean_y=0)

        Ps = self.cfft2(patch)  #TODO
        self.A = np.mean([self.A * (1-self.learning_rate) + np.conjugate(self.C) * P * self.learning_rate for P in Ps], axis=0)
        self.B = np.mean([self.B * (1-self.learning_rate) + np.conjugate(P) * P * self.learning_rate for P in Ps], axis=0)
        self.M = self.A / self.B

        return patch

