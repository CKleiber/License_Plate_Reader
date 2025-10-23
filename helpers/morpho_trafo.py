import numpy as np

def dialation(img):
    # dialation with 3x3 square structuring element
    height, width = img.shape

    kernel_size = (min(height, width) // 65 + 1) if ((min(height, width) // 65) % 2 == 0) else (min(height, width) // 65) # 65 is arbitrary
    img_padded= np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='edge')

    # create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(img_padded, (kernel_size, kernel_size))
    windows = windows.reshape(img.shape[0], img.shape[1], kernel_size, kernel_size)

    # apply dialation: if any value in the window is 1, set to 1, else set to 0
    img_dilated = np.max(windows, axis=(2, 3))

    return img_dilated

def erosion(img):
    # erosion with 3x3 square structuring element
    height, width = img.shape

    kernel_size = (min(height, width) // 65 + 1) if ((min(height, width) // 65) % 2 == 0) else (min(height, width) // 65) # 65 is arbitrary
    img_padded= np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='edge')

    # create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(img_padded, (kernel_size, kernel_size))
    windows = windows.reshape(img.shape[0], img.shape[1], kernel_size, kernel_size)

    # apply erosion: if all values in the window are 1, set to 1, else set to 0
    img_eroded = np.min(windows, axis=(2, 3))

    return img_eroded

def closing(img):
    # closing = dilation followed by erosion
    img_dilated = dialation(img)
    img_closed = erosion(img_dilated)
    return img_closed
