from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from engine.fen import to_matrix, matrix_to_fen

def get_figures(model, image_path):
    # detect figures
    image = Image.open(image_path)
    results = model(image)

    objects = []
    for detection in results:
        for data in detection.boxes.data:    
            # (x1,y1), Pw1
            objects.append([(int(data[0]), int(data[1])), detection.names[int(data[5])]])
    # print("detected_objects:", len(objects))
    return objects

# detect board position
def get_board_position(image_path, figures):
    
    # 1. detect lines
    # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html

    # Load image
    image = cv2.imread(image_path) #'datasets/test/images/0000.png'
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image slightly to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 100)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100, 50, 50)  # Adjust parameters as needed

    # Draw detected lines on the original image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            length = 640
            x1 = int(x0 + length * (-b))
            y1 = int(y0 + length * (a))
            x2 = int(x0 - length * (-b))
            y2 = int(y0 - length * (a))

            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # show line
    # cv2.imshow('Squares', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 2. detect squares
    # https://learnopencv.com/contour-detection-using-opencv-python-c/
    cv2.imwrite('tmp.jpg', image)
    image = cv2.imread('tmp.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # draw contours on the original image
    cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    
    # see the results
    # cv2.imshow('None approximation', image_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Filter contours
    c = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # A square has 4 vertices
            area = cv2.contourArea(contour)
            if area < 500 or area > 2000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            c.append([x, y, contour])

    # TODO order squares
    c = list(map(lambda x: x[2], sorted(c, key=lambda x: [x[1], x[0]], reverse=False)))
    
    # Draw and index squares
    for i, square in enumerate(c):
        cv2.drawContours(image, [square], -1, (0, 255, 0), 1)
        x, y, w, h = cv2.boundingRect(square)
        cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 1000), 1)


    # Display the result
    # cv2.imshow('Squares', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 3. find positions of figures
    result = []
    for contour in c:
        name = ' '
        for figure in figures:
            # Check if point is inside the contour
            # print(figure[0])
            dist = cv2.pointPolygonTest(contour, figure[0], False)
            if dist >= 0:
                name = figure[1]
                break
        
        result.append(name)

    return result

class Detector:

    def __init__(self):
        self.model = YOLO('pt/best.pt')

    # detect move from image and generate fen
    def detect(self, image_path=''):
        image_path = 'datasets/test/images/0000.png'        
        figures = get_figures(self.model, image_path)
        positions = get_board_position(image_path, figures)
        matrix = to_matrix(positions)
        fen = matrix_to_fen(matrix)
        print(fen)
        return fen