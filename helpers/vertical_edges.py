import numpy as np

# otsu's thresholding helper function from https://en.wikipedia.org/wiki/Otsu%27s_method
def otsu_intraclass_variance(image, threshold):
    """
    Otsu's intra-class variance.
    If all pixels are above or below the threshold, this will throw a warning that can safely be ignored.
    """
    return np.nansum(
        [
            np.mean(cls) * np.var(image, where=cls)
            #   weight   ·  intra-class variance
            for cls in [image >= threshold, image < threshold]
        ]
    )

def detect_vertical_edges(image):
    # use the Sobel operator to detect vertical edges
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # pad image with 'clamp' mode
    img_padded= np.pad(image, ((1, 1), (1, 1)), mode='edge')

    # create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(img_padded, (3, 3))
    windows = windows.reshape(image.shape[0], image.shape[1], 3, 3)

    # convolve using einsum
    image_vertical_edges = np.einsum('hwij,ij->hw', windows, sobel_x)

    # otsus thresholding approach
    otsu_threshold = min(
        np.linspace(np.min(np.abs(image_vertical_edges)) + 0.001, np.max(np.abs(image_vertical_edges)), 100),
        key=lambda th: otsu_intraclass_variance(np.abs(image_vertical_edges), th))

    # apply the threshold
    image_vertical_edges = np.where(np.abs(image_vertical_edges) >= otsu_threshold, 1, 0)

    return image_vertical_edges