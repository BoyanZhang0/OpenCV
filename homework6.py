import numpy as np
import cv2

# 练习题1
'''
# 滚动条回调函数
def change(x):
    pass

img = cv2.imread("cat.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转化到HSV颜色空间内
# 获取图片轮廓 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化到灰度空间
ret, binary = cv2.threshold(img_gray, 120, 220, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 得到图像轮廓
zero_mask = np.zeros(img_gray.shape, np.uint8)
contour_image = cv2.drawContours(zero_mask, contours, -1, (255, 255, 255), -1)  # 绘制图像轮廓

cv2.namedWindow('image')
# 创建滚动条
cv2.createTrackbar('LowerbH', 'image', 0, 255, change)
cv2.createTrackbar('LowerbS', 'image', 0, 255, change)
cv2.createTrackbar('LowerbV', 'image', 0, 255, change)
cv2.createTrackbar('UpperbH', 'image', 255, 255, change)
cv2.createTrackbar('UpperbS', 'image', 255, 255, change)
cv2.createTrackbar('UpperbV', 'image', 255, 255, change)
while(1):
    lowerbH = cv2.getTrackbarPos('LowerbH', 'image')
    lowerbS = cv2.getTrackbarPos('LowerbS', 'image')
    lowerbV = cv2.getTrackbarPos('LowerbV', 'image')
    upperbH = cv2.getTrackbarPos('UpperbH', 'image')
    upperbS = cv2.getTrackbarPos('UpperbS', 'image')
    upperbV = cv2.getTrackbarPos('UpperbV', 'image')
    img_mask = cv2.inRange(img_hsv, (lowerbH,lowerbS,lowerbV),(upperbH,upperbS,upperbV))  # 得到目标颜色的二值图像
    img_result = cv2.bitwise_and(img, img, mask=img_mask)  # 输入图像与输入图像在掩模条件下按位与，得到掩模范围内的原图像

    cv2.imshow('image', img_result)  # 图像分割后的图像
    cv2.imshow('contour', contour_image)  # 图像轮廓
    cv2.imshow('origin', img)  # 原图像

    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
'''
# 练习题2
'''
# 滚动条回调函数
def change(x):
    pass

img = cv2.imread("cat.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 得到灰度图

cv2.namedWindow('image')
# 创建滚动条
cv2.createTrackbar('value', 'image', 0, 255, change)
cv2.createTrackbar('type', 'image', 0, 5, change)

while(1):
    img_copy1 = img.copy()  # 用于显示轮廓外接矩形
    img_copy2 = img.copy()  # 用于显示轮廓凸包
    value = cv2.getTrackbarPos('value', 'image')
    type = cv2.getTrackbarPos('type', 'image')
    if type == 0:  # 滚动条值为0时，选用THRESH_BINARY
        t1, thd = cv2.threshold(img_gray, value, 255, cv2.THRESH_BINARY)
    elif type == 1:  # 滚动条值为1时，选用THRESH_BINARY_INV
        t1, thd = cv2.threshold(img_gray, value, 255, cv2.THRESH_BINARY_INV)
    elif type == 2:  # 滚动条值为2时，选用THRESH_TRUNC
        t1, thd = cv2.threshold(img_gray, value, 255, cv2.THRESH_TRUNC)
    elif type == 3:  # 滚动条值为3时，选用THESH_TOZERO
        t1, thd = cv2.threshold(img_gray, value, 255, cv2.THRESH_TOZERO)
    elif type == 4:  # 滚动条值为4时，选用THRESH_TOZERO_INV
        t1, thd = cv2.threshold(img_gray, value, 255, cv2.THRESH_TOZERO_INV)
    elif type == 5:  # 滚动条值为5时，选用Otsu方法，此时value=0不随value滚动条变化
        t1, thd = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 过滤噪声
    thd = cv2.medianBlur(thd, 3)
    # 得到选择后图像的轮廓
    contours, hierarchy = cv2.findContours(thd, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    zero_mask = np.zeros(img_gray.shape, np.uint8)
    contour_image = cv2.drawContours(zero_mask, contours, -1, (255, 255, 255), -1)

    # 轮廓外接矩形 正矩形,不可倾斜
    x, y, w, h = cv2.boundingRect(contour_image)
    img_rect = cv2.rectangle(img_copy1, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # 轮廓凸包
    hull = cv2.convexHull(contours[0])  # 寻找凸包，得到凸包的角点
    cv2.polylines(img_copy2, [hull], True, (255, 255, 0), 2)  # 绘制凸包
    #----------凸包效果在我自己使用的cat图片中效果并不好但在hand图片中效果较好----------

    cv2.imshow('image', thd)  # 显示滚动条选择后的图片
    cv2.imshow('rect_contour', img_rect)  # 显示轮廓长方形
    cv2.imshow('hull_contour', img_copy2)  # 显示轮廓凸包
    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
'''
# 练习题3
'''
img = cv2.imread("cat.jpg")
x, y = img.shape[:2]
background = cv2.imread("watermark.png")
background = cv2.resize(background, dsize=(y, x), interpolation=cv2.INTER_LINEAR)  # 使新的背景图与原图片大小一致

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转化到HSV颜色空间内
# 设置分割背景的颜色上下阈值
lower_yellow = np.array([0, 50, 50])
upper_yellow = np.array([50, 255, 255])

img_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)  # 得到目标颜色的二值图像
mask = cv2.bitwise_not(img_mask)  # 对img_mask取反
background = cv2.bitwise_and(background, background, mask=mask)  # 将背景中mask取反对应的值变为0
img_result = cv2.bitwise_and(img, img, mask=img_mask)  # 输入图像与输入图像在掩模条件下按位与，得到掩模范围内的原图像
final_img = background + img_result  # 相加得到换背景后的图像。

cv2.imshow('image', final_img)
cv2.waitKey()
cv2.destroyAllWindows()
'''
# 练习题4
'''
# 滚动条回调函数
def change(x):
    pass

img = cv2.imread("human.jpg")
rows, cols = img.shape[:2]
cv2.namedWindow('image')

#  采用交互式前景提取人像
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 400, 450)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

img_human = img * mask2[:, :, np.newaxis]
img_copy = img_human.copy()  # 用于替换背景颜色时进行判断是否为背景

# 创建滚动条
cv2.createTrackbar('r', 'image', 255, 255, change)
cv2.createTrackbar('g', 'image', 255, 255, change)
cv2.createTrackbar('b', 'image', 255, 255, change)

while(1):
    r = cv2.getTrackbarPos('r', 'image')
    g = cv2.getTrackbarPos('g', 'image')
    b = cv2.getTrackbarPos('b', 'image')
    for i in range(rows):
        for j in range(cols):
            if img_copy[i, j][0] == 0 & img_copy[i, j][1] == 0 & img_copy[i, j][2] == 0:
                img_human[i, j] = (b, g, r)
    cv2.imshow('image', img_human)  # 替换背景后的图像
    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
'''
# 练习题5
'''
img = cv2.imread("cat.jpg")

result = 0  # 用于存储最后的结果图片
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
mask = np.zeros(img.shape[:2], dtype=np.uint8)  # 原图mask
# 鼠标选择区域后，按下ENTER进行目标效果预览，如不满意可继续使用鼠标选择
while(1):
    x_min, y_min, w, h = cv2.selectROI('input', img, False)  # 选择矩形框
    if w == 0 | h == 0:  # 如果选择完成，则不会产生下一个矩形框，此时w和h等于0，故直接停止循环
        break
    rect = (int(x_min), int(y_min), int(w), int(h))  # 包括前景的矩形，格式为(x,y,w,h)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')  # 提取前景和可能的前景区域
    result = cv2.bitwise_and(img, img, mask=mask2)  # 得到目标图片

    cv2.imshow('result', result)
    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 将图片背景透明化
result.astype('uint8')  # 由于result为float64型的array，opencv不支持，故将result转化为opencv支持的类型
img_final = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)  # 将bgr转化为bgra图像
tmp = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # 将bgr转化为灰度图

b, g, r, a = cv2.split(img_final)
_, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
rgba = [b, g, r, alpha]
img_final = cv2.merge(rgba)

cv2.imwrite('result.png', img_final)  # 保存只包含前景目标的透明图像
cv2.waitKey()
cv2.destroyAllWindows()
'''
# 练习题6
def draw_white(event, x, y, flags, param):  # 鼠标回调函数
    global ix, iy, drawing, img_copy, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            cv2.circle(img_copy, (x, y), 3, (255, 255, 255), -1)  # 用于显示给用户看哪些mask被改为1
            cv2.circle(mask, (x, y), 3, 1, -1)  # 将mask的值变为1
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False

img = cv2.imread("cat.jpg")
img_copy = img.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)  # 原图mask
drawing = False
ix, iy = -1, -1
result = 0  # 用于存储最后的结果图片
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_white)
while(1):
    cv2.imshow('image', img_copy)
    cv2.imshow('origin', img)  # 原图
    cv2.imshow('result', result)  # 目标图片
    if cv2.waitKey(1) & 0xFF == 13:  # 按ENTER进行目标图片预览
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, mode=cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) + (mask == 0), 0, 1).astype('uint8')  # 提取前景和可能的前景区域
        result = img * mask2[:, :, np.newaxis]

    if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
        break

# 将图片背景透明化
result.astype('uint8')  # 由于result为float64型的array，opencv不支持，故将result转化为opencv支持的类型
img_final = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)  # 将bgr转化为bgra图像
tmp = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # 将bgr转化为灰度图

b, g, r, a = cv2.split(img_final)
_, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
rgba = [b, g, r, alpha]
img_final = cv2.merge(rgba)

cv2.imwrite('mask_result.png', img_final)  # 保存只包含前景目标的透明图像
cv2.waitKey()
cv2.destroyAllWindows()
