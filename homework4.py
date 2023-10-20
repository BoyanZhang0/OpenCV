import numpy as np
import cv2

def img_embossing():
    global size, dir

def change_way(x):
    global way
    way = cv2.getTrackbarPos('way', 'image')

def change_bcsc(x):
    global brightness, contrast, saturation, color_temperature
    brightness = cv2.getTrackbarPos('Brightness', 'image')
    contrast = cv2.getTrackbarPos('Contrast', 'image')
    saturation = cv2.getTrackbarPos('Saturation', 'image')
    color_temperature = cv2.getTrackbarPos('Color temperature', 'image')

# 练习题1
size = 1
dir = 1
'''
# 练习题2
way = 1
img2 = cv2.cvtColor(cv2.imread("cat.jpg"), cv2.COLOR_BGR2GRAY)
img2_copy = cv2.cvtColor(cv2.imread("cat.jpg"), cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image')
cv2.createTrackbar('way', 'image', 0, 3, change_way)
while(1):
    way = cv2.getTrackbarPos('way', 'image')
    if way == 0:  # Laplacian边缘检测
        img2 = img2_copy.copy()
        laplacian = cv2.Laplacian(img2, -1)
        img = np.hstack((laplacian, img2_copy))
        cv2.imshow('image', img)
    elif way == 1:  # Canny边缘检测
        img2 = img2_copy.copy()
        canny = cv2.Canny(img2, 50, 240)
        img = np.hstack((canny, img2_copy))
        cv2.imshow('image', img)
    elif way == 2:  # Soble边缘检测,其中dx=0，dy=1
        img2 = img2_copy.copy()
        sobel = cv2.Sobel(img2, -1, 0, 1)
        img = np.hstack((sobel, img2_copy))
        cv2.imshow('image', img)
    elif way == 3:  # # Soble边缘检测,其中dx=1，dy=0
        img2 = img2_copy.copy()
        sobel = cv2.Sobel(img2, -1, 1, 0)
        img = np.hstack((sobel, img2_copy))
        cv2.imshow('image', img)

    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
'''
#练习题3
brightness, contrast, saturation, color_temperature = 1, 1, 1, 1
img3 = cv2.imread("cat.jpg")
img3_copy = img3.copy()

cv2.namedWindow('image')
cv2.createTrackbar('Brightness', 'image', 10, 15, change_bcsc)
cv2.createTrackbar('Contrast', 'image', 0, 5, change_bcsc)
cv2.createTrackbar('Saturation', 'image', 0, 3, change_bcsc)
cv2.createTrackbar('Color temperature', 'image', 0, 3, change_bcsc)
while(1):
    img3 = img3_copy.copy()
    # 亮度调整
    img3 = np.power(img3, brightness * 0.1)
    # 对比度调整
    img3 = float(contrast) * img3
    img3[img3 > 255] = 255  # 大于255要截断为255
    img3 = np.round(img3).astype(np.uint8)  # 数据类型的转换

    img = np.hstack((img3, img3_copy))
    cv2.imshow("image", img)
    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
#练习题4
