import numpy as np
import cv2

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, size, img1
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            cv2.circle(img1, (x, y), size, (255, 255, 255), -1)
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False

def changeSize(x):
    size = cv2.getTrackbarPos('size', 'image')
    return size

drawing = False
ix,iy = -1,-1
size = 3
img1 = cv2.imread("cat.jpg")
img2 = cv2.imread("cat.jpg")

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
cv2.createTrackbar('size', 'image', 1, 10, changeSize)
while(1):
    img = np.hstack((img1, img2))
    cv2.imshow('image', img)
    size = cv2.getTrackbarPos('size', 'image')
    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
