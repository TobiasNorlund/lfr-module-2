import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.stats import multivariate_normal
from cvl.dataset import BoundingBox
import cv2
from cvl.image_io import crop_patch
from mosse import MOSSETrackerGrayscale


def get_multichannel_tracker(args):
    return MOSSEMultiChannel(learning_rate=args.learning_rate, std=args.std)


def default_descriptor(image):
    return image

def gradient_descriptor(image):
    image = np.sum(image, axis=2) / 3
    laplacian = cv2.Laplacian(image,cv2.CV_64F)[:,:,np.newaxis]
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)[:,:,np.newaxis]
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)[:,:,np.newaxis]
    return np.abs(np.concatenate([sobelx, sobely, laplacian], axis=2))

def get_feature_descriptor(feature_descriptor):
    if feature_descriptor is None:
        return default_descriptor
    elif feature_descriptor == 'gradient':
        return gradient_descriptor
    else:
        raise ValueError("The passed feature descriptor: {} is not implemented".format(feature_descriptor))

class MOSSEMultiChannel:
    def __init__(self, std=1.0, learning_rate=0.1, feature_descriptor=None):
        self.learning_rate = learning_rate
        self.width = None
        self.height = None
        self.std = std
        self.descriptor = get_feature_descriptor(feature_descriptor)
        self.region = None
        self.region_center = None
        self.mosse_trackers = []


    def start(self, image, region):
        """
            Initialize multiple MOSSE trackers, one per channel.
        """

        self.region = region
        self.region_center = (region.height // 2, region.width // 2)

        i_h, i_w, _ = image.shape
        d_image = self.descriptor(image)
        d_h, d_w, d_channels = d_image.shape

        d_image = cv2.resize(d_image, (i_h, i_w), interpolation=cv2.INTER_AREA)

        for channel in range(d_channels):
            self.mosse_trackers.append(MOSSETrackerGrayscale(std=self.std, learning_rate=self.learning_rate))
            self.mosse_trackers[-1].start(d_image[:,:,channel], region) 

        xy = np.mgrid[0:self.region.height, 0:self.region.width].reshape(2,-1).T
        gaussian = multivariate_normal(mean=np.array(self.region_center), cov=np.eye(2)*self.std**2).pdf(xy)
        gaussian = np.array(gaussian).reshape(self.region.height, self.region.width)
        self.last_response = gaussian

    def detect(self, image):
        """
        Find the object's new position in image, using current model M
        """
        i_h, i_w, _ = image.shape
        d_image = self.descriptor(image)
        d_image = cv2.resize(d_image, (i_h, i_w), interpolation=cv2.INTER_AREA)

        responses = [tracker.compute_response(d_image[:,:,i]).last_response for i, tracker in enumerate(self.mosse_trackers)]
        accum_response = np.sum(responses, axis=0)
        self.last_response = accum_response

        r, c = np.unravel_index(np.argmax(accum_response), accum_response.shape) # Try taking the expected position as well

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        if self.region.intersection_box(BoundingBox("tl-size", 0, 0, image.shape[1], image.shape[0])).area() > 0.0:
            self.region.ypos += r_offset
            self.region.xpos += c_offset

            for tracker in self.mosse_trackers:
                tracker.region.ypos += r_offset
                tracker.region.xpos += c_offset
            
        return self.region

    def update(self, image):
        """
        Re-fit model M using new object position found in self.region (from detection step)
        """
        for tracker in self.mosse_trackers:
            tracker.update(image)
            
    def get_filter(self, image):
        region = self.region
        
        if len(image.shape)==3 : image = np.sum(image, axis=2) / 3
        patch = crop_patch(image, region)
        window = patch
        
        filters = [np.abs(ifft2(tracker.M)) for tracker in self.mosse_trackers]
        filt = np.sum(filters, axis=0)

        response = np.abs(self.last_response)
        return window, filt, response