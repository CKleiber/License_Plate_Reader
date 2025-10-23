import numpy as np

# contour class
class contour:
    def __init__(self, label, initial_pixel):
        # label is an integer
        self.label = label

        # pixels is a list of (y, x) tuples
        self.pixels = [initial_pixel]

        # continuously update area
        self.area = self._area()

    def add_pixel(self, pixel):
        self.pixels.append(pixel)
        self.area = self._area()

    # area update function
    def _area(self):
        return len(self.pixels)
    
    def _centre(self):
        return tuple(np.mean(np.array(self.pixels), axis=0).astype(int))
    
    # calculate the bounding box of all pixels in the contour
    def _calculate_bounding_box(self):
        points = np.array(self.pixels)
        min_y, min_x = np.min(points, axis=0)
        max_y, max_x = np.max(points, axis=0)

        self.bounding_box = ((min_y, min_x), (max_y, max_x))

    # aspect ratio of the bounding box (width / height)
    def calculate_aspect_ratio(self):
        if not hasattr(self, 'bounding_box'):
            self._calculate_bounding_box()
        (min_y, min_x), (max_y, max_x) = self.bounding_box
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        self.aspect_ratio = width / height
        return self.aspect_ratio


# find contours in a binary image
def find_contours(binary_image):
    height, width = binary_image.shape
    label = 1
    labels = np.zeros((height, width), dtype=int)
    contours = []

    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 1 and labels[y, x] == 0:
                # new contour found
                new_contour = contour(label, (y, x))
                contours.append(new_contour)
                labels[y, x] = label

                # perform BFS to find all connected pixels
                queue = [(y, x)]
                while queue:
                    cy, cx = queue.pop(0)
                    for ny in range(max(0, cy-1), min(height, cy+2)):
                        for nx in range(max(0, cx-1), min(width, cx+2)):
                            if binary_image[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = label
                                new_contour.add_pixel((ny, nx))
                                queue.append((ny, nx))
                                
                # increment label for next contour
                label += 1

    return contours
