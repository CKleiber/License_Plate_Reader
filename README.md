This project has the goal of building an algorithm reading a (European) license plate from an image and giving an output string with the respective probability. The aim is to implement this using only the numpy and matplotlib packages, except for the character recognition, where PyTorch is also used to build a CNN.

---

# Outline of the Code

## License Plate Detection

### 1. Image Preprocessing

In a series of steps, the input image is converted from RGB colours to grey scale. On this grey scale image, the vertical edges are detected using a convolution using the **Sobel kernel**. Afterwards, a **closing morphological transformation** is applied on the edges. The image is then binarised using **Otsu's threshold**.

### 2. Finding Contours

Each connecting series of signal pixels is consolidated into a contour object. Each of these contours has a **bounding box** and an area of all pixels. The area is used to filter out contours which are too large or too small to be a license plate. Of the remaining contours, the one whose bounding box has an **aspect ratio** closes to an actual number plate is assumed to be the license palte in the image. The grey scale image is cut using the bounding box and used in the next step.

## Isolating Characters on the License Plate

### 1. Segmentation of the Individual Characters

The isolated grey scaled license plate is simply binarised similarly to above. As it is assumed that the license plates contain black letters on white background, the thresholding is inverted, such that the letters are now white (signal) in front of a black background. 

In the binarised image, contours are again searched and filtered according to an expected range of sized and aspect ratios. Each characters contour is used to gain the bounding box, isolating each individual character.

### 2. Rescaling

These individual characters are then rescaled, such that they fit onto a $28 \times 28$ canvas, compatible with a **EMNIST**-trained **CNN**.

## The Convolutional Neural Network

### 1. Training the Model

To recognise the characters, a simple **CNN** is trained on the **EMNIST** data set [1]  (filtered for digits and capital letters) in a first step, and after convergence **fine tuned** on a personally collected data set using European license plate fonts, found [here](https://github.com/CKleiber/ELPF-data).

### 2. Applying the Model

Each character of the license plate is then evaluated with the model to get a **softmax** probability of each potential character. These probabilities are then used to combine all possible license plates and assign conditional probabilities. The most likely license plates are returned as a string with the respective probabilities.


# Conclusion

The algorithm works fine for the provided images. However, other images might require fine tuning of the contour finding, depending on the light, license plate size, surroundings, and so on. 

The CNN is doing well in finding the correct digit or letter most of the time, but in this mixture of fonts, the letter 'O' and the digit '0' are sometimes identical, just as well as the letter 'I' and the digit '1'. These are the most mix-ups, which are arguably just as likely for a human to get wrong.

---


[1] Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from [http://arxiv.org/abs/1702.05373](http://arxiv.org/abs/1702.05373)