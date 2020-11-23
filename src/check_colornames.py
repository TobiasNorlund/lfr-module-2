from cvl.features import colornames_image

import numpy as np

red_image = np.zeros((128, 128, 3), dtype=np.uint8)
red_image[:, :, 0] = 255

colornames_image = colornames_image(red_image, 'probability')

print(colornames_image.shape)