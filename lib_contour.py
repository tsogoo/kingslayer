import cv2
import numpy as np


def get_enhanced_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 100, 250)

    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw non-linear contours
    for contour in contours:
        # Approximate contour to reduce the number of points
        epsilon = 0.0001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter out contours with less than 5 points (linear shapes)
        # if len(approx) > 4:
        # cv2.drawContours(enhanced, [contour], -1, (255, 255, 255), 1)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

    # Draw contours on the original image

    # cv2.drawContours(enhanced, contours, -1, (155, 155, 155), 1)
    return enhanced


def get_contoured_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 100, 250)

    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw non-linear contours
    for contour in contours:
        # Approximate contour to reduce the number of points
        epsilon = 0.0001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter out contours with less than 5 points (linear shapes)
        if len(approx) > 4:
            cv2.drawContours(enhanced, [contour], -1, (255, 255, 255), 1)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

    # Draw contours on the original image

    cv2.drawContours(enhanced, contours, -1, (155, 155, 155), 1)
    return enhanced


def get_noisy_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    noisy = np.zeros(gray.shape, np.uint8)

    cv2.randn(noisy, 0, 3)

    enhanced = cv2.add(gray, noisy)

    return enhanced


def get_blurry_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply bilateral filter
    # enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # apply blur
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return enhanced
