import cv2

index = '1610'
image = cv2.imread("datasets/train/images/"+index+".png")
with open('datasets/train/labels/'+index+'.txt', 'r') as file:
    for line in file:
        l = line.strip()
        c = l.split()
        x1 = round(float(c[1])*640)
        y1 = round(float(c[2])*640)
        x2 = x1 + round(float(c[3])*640)
        y2 = y1 + round(float(c[4])*640)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(image, c[0], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# show line
cv2.imshow('Squares', image)
cv2.waitKey(0)
cv2.destroyAllWindows()