from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from engine.fen import to_matrix, matrix_to_fen
from functools import reduce

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

def capture_image(image_path):
    # Initialize the camera
    camera = cv2.VideoCapture("/dev/video2")  # 0 represents the default camera, change it if you have multiple cameras

    # Check if the camera is opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a frame
    ret, frame = camera.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        return

    # Save the captured frame as an image file
    cv2.imwrite(image_path, frame)

    # Release the camera
    camera.release()

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
    cv2.imshow('Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2. detect squares
    # https://learnopencv.com/contour-detection-using-opencv-python-c/
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # draw contours on the original image
    # cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    
    # see the results
    cv2.imshow('Squares', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    
    cv2.drawContours(image, [c[0], c[-1]], -1, (0, 255, 0), 2)

    # Draw and index squares
    for i, square in enumerate(c):
        cv2.drawContours(image, [square], -1, (0, 255, 0), 1)
        x, y, w, h = cv2.boundingRect(square)
        cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 1000), 1)


    # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

    # 4. rotate image

    # # Get the minimum area rectangle bounding the contour
    # rect = cv2.minAreaRect(c[0])
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)

    # # Calculate the angle of rotation
    # angle = rect[2]

    # # Rotate the image by the calculated angleqww
    # rows, cols = image.shape[:2]
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    # rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # # Display the result
    # cv2.imshow('Squares', rotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result


def detect(board_pt, figure_pt, image_path:str):

    # 1. detect board
    board_results = board_pt.predict(image_path, save=False, imgsz=640, conf=0.2)
    img = cv2.imread(image_path)
    img_copy = img.copy()
    
    margin = 80
    for i, result in enumerate(board_results):
        x1, y1, x2, y2 = result.boxes.xyxy[i]
        y1 = y1
        y2 = y2
        x1 = x1
        x2 = x2
        cv2.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1
        )
        cv2.putText(
            img, result.names[int(result.boxes.cls[i])], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    ox1, oy1, ox2, oy2 = x1-margin, y1-margin, x2+margin, y2+margin
    cv2.imwrite('01_board.png', img)

    # 2. detect figures
    chess_results = figure_pt.predict(image_path, save=False, imgsz=864, conf=0.3)
    img = img_copy.copy()
    for i, result in enumerate(chess_results):
        x1, y1, x2, y2 = result.boxes.xyxy[i]
        cv2.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1
        )
        cv2.putText(
            img, result.names[int(result.boxes.cls[i])], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    cv2.imwrite('02_figures.png', img)

    # 3. find corners of board, calculate perspective transform matrix
    img = img_copy.copy()
    img = img[int(oy1):int(oy2),int(ox1):int(ox2)]    # crop
    img_crop = img.copy()

    # detect edges using Canny
    blur = cv2.GaussianBlur(img, (3,3), 1)
    cv2.imwrite('03_blur.png', blur)
    # edges = cv2.Canny(img, 350, 400)
    edges = cv2.Canny(blur, 150, 200)
    cv2.imwrite('04_edges.png', edges)

    h,w,_ = img.shape
    gap = max(w/8,h/8)
    line_length = min(h,w)*0.6
    lines = []
    for threshold in range(100, 300, 50):
        l = cv2.HoughLinesP(edges, 1, np.pi/720,
                            threshold=threshold, minLineLength=line_length,
                            maxLineGap=gap)
        if l is not None:
            lines.extend(l)
    
    # draw detected lines
    img_t = img.copy()
    img_t[:] = (255,255,255) # fill with white
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0,0,0), 1)
        cv2.line(img_t, (x1, y1), (x2, y2), (0,0,0), 2)
    cv2.imwrite('05_line.png', img)
    cv2.imwrite('06_line_only.png', img_t)

    # find contour using drawed lines
    img = cv2.cvtColor(img_t, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('07_threshed.png', thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    img_1 = img_crop.copy()
    img_2 = img_1.copy()
    img_2[:] = (255,255,255)
    img_3 = img_1.copy()
    img_3[:] = (255,255,255)
    c = []
    cpts = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # A square has 4 vertices
            x, y, w, h = cv2.boundingRect(contour)
            c.append(contour)
            cpts.append(approx)
            cv2.rectangle(img_1, (x,y), (x+w,y+h), (0,0,0), 2)
    cv2.drawContours(img_crop, c, -1, (0,0,0), 2)
    cv2.imwrite('08_contours.png', img_crop)
    cv2.imwrite('09_contour_bb.png', img_1)
    cv2.drawContours(img_2, c, -1, (0,0,0), 2)
    cv2.imwrite('10_contours.png', img_2)

    # find 4 corner contours

    h,w,_ = img_2.shape
    pts = []
    pts.extend(map(lambda x: x[0][0], cpts))
    pts.extend(map(lambda x: x[1][0], cpts))
    pts.extend(map(lambda x: x[2][0], cpts))
    pts.extend(map(lambda x: x[3][0], cpts))
    p1 = min(pts, key=lambda p: p[0]**2+p[1]**2)            #top left
    p2 = min(pts, key=lambda p: (p[0]-w)**2+p[1]**2)        #top right
    p3 = min(pts, key=lambda p: (p[0])**2+(p[1]-h)**2)      #bottom left
    p4 = min(pts, key=lambda p: (p[0]-w)**2+(p[1]-h)**2)    #bottom right

    cv2.line(img_3, p1, p2, (0,0,0), 2)
    cv2.line(img_3, p2, p4, (0,0,0), 2)
    cv2.line(img_3, p4, p3, (0,0,0), 2)
    cv2.line(img_3, p1, p3, (0,0,0), 2)
    cv2.imwrite('11_final.png', img_3)

    # 4. do transform
    import numpy
    p = numpy.float32([p1, p2, p3, p4])
    min_x = min(p, key= lambda x: x[0])[0]
    min_y = min(p, key= lambda x: x[1])[1]
    max_x = max(p, key= lambda x: x[0])[0]
    max_y = max(p, key= lambda x: x[1])[1]
    width = max_x-min_x
    height = max_y-min_y
    src = p
    dst = numpy.float32([(0, 0),(width, 0),(0, height), (width, height)])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img_1, M, (int(max_x - min_x), int(max_y - min_y)), flags=cv2.INTER_LINEAR)
    cv2.imwrite('12_perspective.png', out)


def detect2(board_pt, image_path:str, is_debug:bool=False, idx:int=1, output:bool=False, from_step:int=11):
    
    # 1. detect board, crop image
    # board_results = board_pt.predict(image_path, save=False, imgsz=640, conf=0.2)
    image = cv2.imread(image_path)
    image_copy = image.copy()
    if is_debug and from_step <= 0:
        cv2.imshow('orig', image)
        cv2.waitKey(0)

    h,w,_ = image.shape

    margin = 80
    # for i, result in enumerate(board_results):
    #     x1, y1, x2, y2 = result.boxes.xyxy[i]
    #     print(x1, y1, x2, y2)
    # ox1, oy1, ox2, oy2 = x1-margin, y1-margin, x2+margin, y2+margin
    # if ox1 < 0:
    #     ox1 = 0
    # if oy1 < 0:
    #     oy1 = 0
    # if ox2 > w:
    #     ox2 = w
    # if oy2 > h:
    #     oy2 = h

    # if is_debug:
    #     cv2.rectangle(image_copy,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,0),2)
    #     cv2.imshow('board_bb', image_copy)
    #     cv2.waitKey(0)

    # image = image[
    #     int(oy1):int(oy2),
    #     int(ox1):int(ox2)
    # ]    # crop
    if is_debug and from_step <= 1:
        cv2.imshow('cropped', image)
        cv2.waitKey(0)

    # 2. detect edge, sharpen, threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if is_debug and from_step <= 2:
        cv2.imshow('grayed', gray)
        cv2.waitKey(0)
    
    blur = cv2.medianBlur(gray, 5)
    if is_debug and from_step <= 3:
        cv2.imshow('blured', blur)
        cv2.waitKey(0)

    # edge detection
    sharpen_kernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    if is_debug and from_step <= 4:
        cv2.imshow('sharpen_edge', sharpen)
        cv2.waitKey(0)

    # Threshold
    _, thresh = cv2.threshold(sharpen, 150, 250, cv2.THRESH_BINARY_INV)
    if is_debug and from_step <= 5:
        cv2.imshow('threshed', thresh)
        cv2.waitKey(0)

    # morph /remove small gaps/
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    if is_debug and from_step <= 6:
        cv2.imshow('closed', close)
        cv2.waitKey(0)
    
    # morph /close gaps/
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=3)
    if is_debug and from_step <= 7:
        cv2.imshow('opened', open)
        cv2.waitKey(0)

    _, thresh = cv2.threshold(open, 150, 250, cv2.THRESH_BINARY_INV)
    if is_debug and from_step <= 8:
        cv2.imshow('threshed_inv', thresh)
        cv2.waitKey(0)

    # 3. Find largest contour, its corner points
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [contour], -1, (0,0,0), 3)
    if is_debug and from_step <= 9:
        cv2.imshow('detected', image)
        cv2.waitKey(0)
    
    p = []
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) != 4:
        raise Exception("contour has no 4 corners")
    for a in approx:
        p.append(a[0])
    
    # 4. perspective transform
    p = sort_points(p)
    min_x = min(p, key= lambda x: x[0])[0]
    min_y = min(p, key= lambda x: x[1])[1]
    max_x = max(p, key= lambda x: x[0])[0]
    max_y = max(p, key= lambda x: x[1])[1]
    width = max_x-min_x
    height = max_y-min_y
    size = max(width, height)
    src = np.float32(p)
    dst = np.float32([(0, 0),(size, 0),(size, size),(0, size)])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(image, M, (int(size), int(size)), flags=cv2.INTER_LINEAR)
    if is_debug and from_step <= 10:
        cv2.imshow('perspective', out)
        cv2.waitKey(0)

    # 5. Find squares
    im = out.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if is_debug and from_step <= 11:
        cv2.imshow('square_grayed', gray)
        cv2.waitKey(0)
    
    blur = cv2.medianBlur(gray, 5)
    if is_debug and from_step <= 11:
        cv2.imshow('square_blured', blur)
        cv2.waitKey(0)

    # edge detection
    sharpen_kernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    if is_debug and from_step <= 11:
        cv2.imshow('square_sharpen_edge', sharpen)
        cv2.waitKey(0)

    _, thresh = cv2.threshold(sharpen, 150, 250, cv2.THRESH_BINARY_INV)
    if is_debug and from_step <= 11:
        cv2.imshow('square_threshed', thresh)
        cv2.waitKey(0)

    # shrink square to detect contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    close = cv2.erode(thresh, kernel, iterations=2)
    if is_debug and from_step <= 11:
        cv2.imshow('square_erode', close)
        cv2.waitKey(0)
    
    _, thresh = cv2.threshold(close, 150, 250, cv2.THRESH_BINARY_INV)
    if is_debug and from_step <= 11:
        cv2.imshow('square_threshed_inv', thresh)
        cv2.waitKey(0)

    # remove gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    if is_debug and from_step <= 11:
        cv2.imshow('square_opened', open)
        cv2.waitKey(0)

    contours, _ = cv2.findContours(open, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (255,0,0), 2)
    if is_debug and from_step <= 11:
        cv2.imshow('square_contours', im)
        cv2.waitKey(0)

    # 6. merge original and result images into one image
    h1, w1 = image_copy.shape[:2]
    h2, w2 = image.shape[:2]
    h3, w3 = out.shape[:2]
    merged = np.zeros((max(h1,h2,h3),w1+w2+w3,3), np.uint8)
    merged[:h1, :w1,:3] = image_copy
    merged[:h2, w1:w1+w2,:3] = image
    merged[:h3, w1+w2:w1+w2+w3,:3] = out
    if output:
        cv2.imwrite('result/{}.png'.format(idx), merged)

    if is_debug:
        cv2.destroyAllWindows()

def angle_from_centroid(centroid, point):
    return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

def sort_points(points):
    centroid = np.mean(points, axis=0)
    sorted_points = sorted(points, key=lambda point: angle_from_centroid(centroid, point))
    return sorted_points

def create_video():
    imgs = []
    for i in range(1, 1262, 1):
        imgs.append(cv2.imread('result/'+str(i)+'.png'))
    h = max(imgs, key= lambda img: img.shape[0]).shape[0]
    w = max(imgs, key= lambda img: img.shape[1]).shape[1]
    video = cv2.VideoWriter('result.mp4',-1,1,w,h)
    for img in imgs:
        video.write(img)
    video.release()

import time
def detect_from_camera(board_pt, figure_pt, image_path:str):
    while True:
        time.sleep(5)

        if True:
            cap = cv2.VideoCapture("http://192.168.1.10:8080/video")
            ret, image = cap.read()
            if ret:
                cv2.imwrite(image_path, image)
            else:
                print('eee')
            cap.release()

        try:
            detect(board_pt, figure_pt, image_path)
        except Exception as e:
            print("Error:", e)

class Detector:

    def __init__(self):
        self.model = YOLO('pt/best.pt')

    # detect move from image and generate fen
    def detect(self, image_path=''):
        image_path = 'datasets/test/images/0000.png'        
        figures = get_figures(self.model, image_path)
        # figures = []
        # capture_image(image_path)
        positions = get_board_position(image_path, figures)
        matrix = to_matrix(positions)
        fen = matrix_to_fen(matrix)
        print(fen)
        return fen