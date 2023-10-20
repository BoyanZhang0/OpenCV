import numpy as np
import cv2

def Text_input(event, x, y, flags, param):
    global ix, iy, drawing, size, img, text, R, B, G
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        cv2.putText(img, text, (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, size, (R, B, G), 5)


def change(x):
    size = cv2.getTrackbarPos('size', 'image')
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    return size

drawing = False
ix, iy = -1,-1
text = "cat"
size = 3
R, B, G = 0, 0, 0
img = cv2.imread("cat.jpg")

cv2.namedWindow('image')
cv2.setMouseCallback('image', Text_input)
cv2.createTrackbar('R', 'image', 0, 255, change)
cv2.createTrackbar('G', 'image', 0, 255, change)
cv2.createTrackbar('B', 'image', 0, 255, change)
cv2.createTrackbar('size', 'image', 3, 10, change)
while(1):
    cv2.imshow('image', img)
    size = cv2.getTrackbarPos('size', 'image')
    R = cv2.getTrackbarPos('R', 'image')
    G = cv2.getTrackbarPos('B', 'image')
    B = cv2.getTrackbarPos('G', 'image')
    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
