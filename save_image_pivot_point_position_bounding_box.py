# save image with bounding box and pivot point
# Usage: python check_pivot_point_position.py -i <input_image> -o <output_image> -p <pivot_point> -b <bounding_box>
# Example: python check_pivot_point_position.py -i test.jpg -o test_output.jpg -p 100,100 -b 50,50,200,200

import cv2
import numpy as np

import argparse


def check_pivot_point_position(input_image, output_image, width, labels):
    # Read image
    image = cv2.imread(input_image)
    image_copy = image.copy()

    # Draw bounding box
    # x1, y1, x2, y2 = bounding_box
    # cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw pivot point
    for label in labels:
        x1, y1, x2, y2 = (
            int((label[1] - label[3] / 2) * width),
            int((label[2] - label[4] / 2) * width),
            int(label[1] * width) + int(label[3] / 2 * width),
            int(label[2] * width) + int(label[4] / 2 * width),
        )
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if len(label) > 5:
            pivot_x, pivot_y = (
                int((label[5] - label[3] / 2) * width),
                int((label[6] - label[4] / 2) * width),
            )  # pivot point
            cv2.circle(image_copy, (pivot_x, pivot_y), 2, (0, 0, 255), -1)

    # Save image
    cv2.imwrite(output_image, image_copy)


for i in range(20):
    mode = "test"
    input_image = f"cm_datasets/{mode}/images/{i:04d}" + ".png"
    output_image = f"cm_datasets/{mode}/pivot/{i:04d}" + ".png"
    width = 840
    labels = []
    with open(f"cm_datasets/{mode}/labels/{i:04d}.txt") as f:
        # loop all lines
        #
        for line in f:
            print(line.split())
            labels.append(tuple(map(float, line.split())))
    check_pivot_point_position(input_image, output_image, width, labels)
