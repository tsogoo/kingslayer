# load images from a folder and save the contour of the images in another folder

import random
import cv2
import os
import numpy as np

from lib_contour import (
    get_enhanced_image,
    get_contoured_image,
    get_blurry_image,
    get_noisy_image,
)


# Load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # append with filename
            images.append([img, filename])
    return images

    # enhanced = cv2.medianBlur(gray, 2)

    # # Apply thresholding
    # _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Apply morphological operations
    # kernel = np.ones((3, 3), np.uint8)
    # enhanced = cv2.erode(enhanced, kernel, iterations=1)
    # enhanced = cv2.dilate(enhanced, kernel, iterations=1)


# Save the contour of the images in another folder
def save_contour_images(images, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(len(images)):
        image = images[i][0]
        rand = random.random()
        if rand < 0.2:
            image = get_enhanced_image(image)
        elif rand < 0.4:
            image = get_contoured_image(image)
        elif rand < 0.6:
            image = get_blurry_image(image)
        elif rand < 0.8:
            image = get_noisy_image(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(folder + "/" + images[i][1], image)


# Load images from a folder
images = load_images_from_folder("datasets/train_images")

# Save the contour of the images in another folder
save_contour_images(images, "datasets/train/images")
