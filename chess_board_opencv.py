import cv2
import numpy as np


# Load your image
image = cv2.imread("0002.jpg")

# warp perspective inverse example
# define the four corners of a chess board
# in the original image
pts1 = np.float32([[0, 0], [0, 640], [640, 0], [640, 640]])

# define the four corners of the chess board
# in the new image
pts2 = np.float32([[200, 0], [0, 640], [440, 0], [640, 640]])


# calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# warp the perspective coordinates
input_coordinates = np.array([[10, 10], [20, 640], [640, 0], [640, 640]])
output_coordinates = cv2.perspectiveTransform(
    np.array([input_coordinates], dtype="float32"), M
)
print(output_coordinates)

# result = cv2.warpPerspective(image, M, (640, 640))


# cv2.imwrite("result.jpg", result)
