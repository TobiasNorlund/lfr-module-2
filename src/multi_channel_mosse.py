import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from cvl.dataset import BoundingBox
import cv2
from cvl.image_io import crop_patch
import mosse

LAMBDA = 0 # Did a quick test with regularization
def add_cli_arguments(parser):
    group = parser.add_argument_group("MCM", "Multi-channel MOSSE Tracker arguments")
    group.add_argument("--learning-rate", type=float, default=0.1)
    group.add_argument("--std", type=float, default=1.0)
    group.add_argument("--feature-descriptor", type=str, default='gradient')

def get_multichannel_tracker(args):
    return MOSSEMultiChannel(learning_rate=args.learning_rate, std=args.std, feature_descriptor=args.feature_descriptor)

def default_descriptor(image):
    return image

def gradient_descriptor(image):
    image_c = np.copy(image)
    image = np.sum(image, axis=2) / 3
    image = image[:,:,np.newaxis]
    laplacian = cv2.Laplacian(image,cv2.CV_64F)[:,:,np.newaxis]
    sobelx = cv2.Sobel(image,cv2.CV_64F, 1, 0, ksize=5)[:,:,np.newaxis]
    sobely = cv2.Sobel(image,cv2.CV_64F, 0, 1, ksize=5)[:,:,np.newaxis]
    #return np.abs(np.concatenate([image, sobelx, sobely, laplacian], axis=2))
    #return np.abs(np.concatenate([image_c, laplacian], axis=2))
    return np.abs(np.concatenate([image, laplacian], axis=2))


def get_feature_descriptor(feature_descriptor):
    if feature_descriptor is None:
        return default_descriptor
    elif feature_descriptor == 'gradient':
        return gradient_descriptor
    else: 
        raise ValueError("The passed feature descriptor: {} is not implemented".format(feature_descriptor))

class MOSSEMultiChannel:
    def __init__(self, std=1.0, learning_rate=0.1, feature_descriptor='gradient'):
        self.learning_rate = learning_rate
        self.width = None
        self.height = None
        self.std = std
        self.descriptor = get_feature_descriptor(feature_descriptor)
        self.region = None
        self.region_center = None
        self.n_channels = None
        self.As = None
        self.B = None
        self.C = None

    @staticmethod
    def normalize(patch):
        patch = np.log(patch+1)
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        window = mosse.window_func_2d(patch.shape[0], patch.shape[1])
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

    def desc_patch_from_image(self, image, region):
        patch = crop_patch(image, region)
        d_patch = self.descriptor(patch)
        d_patch = cv2.resize(d_patch, (region.width, region.height), interpolation=cv2.INTER_AREA)

        if len(d_patch.shape) == 2:
            d_patch = d_patch[:,:,np.newaxis]

        return d_patch

    def start(self, image, region):

        self.region = region
        self.region_center = (region.height // 2, region.width // 2)
        d_patch = self.desc_patch_from_image(image, region)
        
        self.n_channels = d_patch.shape[2]
        self.C = MOSSEMultiChannel.get_fourier_transformed_gaussian(height=region.height, width=region.width, std=self.std,
                                                                    mean_x=0, mean_y=0)
                                                                    
        Ps = [fft2(MOSSEMultiChannel.normalize(d_patch[:,:,i])) for i in range(self.n_channels)]
        self.As = [np.conjugate(self.C) * P for P in Ps]
        self.B = np.sum([np.conjugate(P) * P + LAMBDA for P in Ps], axis=0)
        self.Ms = [A / self.B for A in self.As]

        # For comparison (similar to Ehsan's implementation)
        self.A = np.sum(self.As, axis=0)
        self.M = self.A / self.B

    def detect(self, image):
        """
            Find the object's new position in image, using current model M
        """
        
        d_patch = self.desc_patch_from_image(image, self.region)
        Ps = [fft2(MOSSEMultiChannel.normalize(d_patch[:,:,i])) for i in range(self.n_channels)]
        
        R = np.sum([np.conjugate(M) * P for M, P in zip(self.Ms, Ps)], axis=0)
        response = np.real(ifft2(R))

        #R = np.sum([np.conjugate(self.M) * P for P in Ps], axis=0)
        #response = ifft2(R)
   
        r, c = np.unravel_index(np.argmax(response), response.shape)

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
        d_patch = self.desc_patch_from_image(image, self.region)
        Ps = [fft2(MOSSEMultiChannel.normalize(d_patch[:,:,i])) for i in range(self.n_channels)]

        self.As = [Aprev * (1-self.learning_rate) + np.conjugate(self.C) * P * self.learning_rate for Aprev, P in zip(self.As, Ps)]
        self.B = np.sum([self.B * (1-self.learning_rate) + (np.conjugate(P) * P + LAMBDA) * self.learning_rate for P in Ps], axis=0)
        self.Ms = [A / self.B for A in self.As]

        self.A = np.sum(self.As, axis=0)
        self.M = self.A / self.B