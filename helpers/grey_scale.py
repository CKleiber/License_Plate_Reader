import numpy as np

# function to convert a color image to grey scale
def grey_scale(img):
    # using luminosity method to convert to grey scale
    grey_img = 0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2]
    return grey_img