import numpy as np

# create gaussian kernel
def gaussian_kernel(kernel_size=5, sigma=1.0):
    # assert kernel size is odd
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    # set up kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel_centre = kernel_size // 2

    # generate grid of (i,j) coordinates between 0 and kernel_size-1
    i, j = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size), indexing='ij')

    # calculate G_ij = exp(-( (i - k_centre)^2 + (j - k_centre)^2 ) / (2*sigma^2))
    kernel = np.exp(-((i - kernel_centre) ** 2 + (j - kernel_centre) ** 2) / (2 * sigma ** 2))

    # reshape kernel to (kernel_size, kernel_size, 1) for broadcasting later
    kernel = kernel.reshape((kernel_size, kernel_size, 1))

    # normalize by calculating alpha
    kernel_sum = np.sum(kernel)
    alpha = 1 / kernel_sum
    kernel *= alpha

    return kernel


# apply gaussian blur to image
def gaussian_blur(img, kernel_size=5, sigma=1.0):
    # apply gaussian blur of the shape G_ij = alpha * exp(-(i^2 + j^2) / (2*sigma^2))
    kernel = gaussian_kernel(kernel_size, sigma)

    # if image has channels (used to reduce noise in color images)
    if img.ndim == 3:
        # pad image with condition 'clamp' (repeat edge values)
        img_padded = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), mode='edge')

        # create sliding windows
        height, width, channels = img.shape
        windows = np.lib.stride_tricks.sliding_window_view(img_padded, (kernel_size, kernel_size, channels))
        windows = windows.reshape(height, width, kernel_size, kernel_size, channels)

        # convolve using einsum
        img_blurred = np.einsum('hwijc,ijc->hwc', windows, kernel)/255
        img_blurred = np.clip(img_blurred, 0, 1)

    # if image is grayscale (used to downsize digits and letters on the license plate for comparability with EMNIST)
    else:
        # pad image with condition 'clamp' (repeat edge values)
        img_padded = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='edge')

        # create sliding windows
        height, width = img.shape
        windows = np.lib.stride_tricks.sliding_window_view(img_padded, (kernel_size, kernel_size))
        windows = windows.reshape(height, width, kernel_size, kernel_size, 1)

        # convolve using einsum
        img_blurred = np.einsum('hwijc,ijc->hwc', windows, kernel)
        img_blurred = np.clip(img_blurred, 0, 1)
    
    return img_blurred