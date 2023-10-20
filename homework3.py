import cv2
import numpy as np

def change(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    min_x, min_y, w, h = cv2.selectROI(img, False)

    b, g, r, a = cv2.split(img)
    a[:min_y, :] = 0
    a[min_y: min_y + h, :min_x] = 0
    a[min_y: min_y + h, min_x + w:] = 0
    a[min_y + h:, :] = 0
    img = cv2.merge([b, g, r, a])
    return img

def picture_merge(event, x, y, flags, param):
    global ix, iy, img, img_change, img_back, size
    if event == cv2.EVENT_MOUSEMOVE:
        ix, iy = x, y
        img = img_back.copy()
        img_size_change = cv2.resize(img_change, (0, 0), fx=(size / 5), fy=(size / 5))
        width, height = img_size_change.shape[:2]
        img[iy: iy + width, ix: ix + height] = img_size_change
def changeSize(x):
    size = cv2.getTrackbarPos('size', 'image')

def polygon_coding(event, x, y, flags, param):
    global area, count
    if event == cv2.EVENT_LBUTTONDOWN:
        area.append([x, y])
        print("选择点：(", x, ",", y, ")")

def insert_black_and_white_picture(event, x, y, flags, param):
    global x1, y1, img3, watermark, img_back3
    if event == cv2.EVENT_MOUSEMOVE:
        x1, y1 = x, y
        img3 = img_back3.copy()
        width, height = watermark.shape[:2]
        img3[y1: y1 + width, x1: x1 + height] = watermark

# 练习题1
print("练习题1如下")
img = cv2.imread("cat.jpg")
img_change = change(img)
cv2.imwrite("cat_change.png", img_change)
print("练习题1结束")

# 练习题2
print("练习题2如下")
ix, iy = -1, -1
size = 1
img = cv2.imread("cat.jpg")
img_back = img.copy()
img_change = cv2.imread("watermark1.png")  # 待插入图片

cv2.namedWindow('image')
cv2.createTrackbar('size', 'image', 1, 5, changeSize)
cv2.setMouseCallback('image', picture_merge)

while 1:
    cv2.imshow('image', img)
    size = cv2.getTrackbarPos('size', 'image')
    if cv2.waitKey(1) & 0xFF == 13:  # Enter
        cv2.imwrite("cat_merge.png", img)
        break
cv2.destroyAllWindows()
print("练习题2结束")

# 练习题3
print("练习题3如下")
area = []
count = 0
img2 = cv2.imread("cat.jpg", 0)
cv2.namedWindow('image')
cv2.setMouseCallback('image', polygon_coding)  # 选择顶点

while 1:
    cv2.imshow('image', img2)
    if cv2.waitKey(1) & 0xFF == 13:  # Enter
        break
cv2.destroyAllWindows()

r, c = img2.shape
mask = np.zeros((r, c), dtype=np.uint8)
area = np.array(area)
cv2.polylines(mask, [area], 1, 255)  # 描绘边缘
cv2.fillPoly(mask, [area], 1)  # 填充
key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
catXorKey = cv2.bitwise_xor(img2, key)
encryptFace = cv2.bitwise_and(catXorKey, mask * 255)
noFace = cv2.bitwise_and(img2, (1 - mask) * 255)
maskFace = encryptFace + noFace

cv2.imshow('image', maskFace)
cv2.waitKey()
cv2.destroyAllWindows()
print("练习题3结束")
# 练习题4
print("练习题4如下")
x1, y1 = -1, -1
size = 1
img3 = cv2.imread("cat.jpg", 0)
img_back3 = img3.copy()

# 得到黑白二值图像
# watermark = cv2.imread("watermark.png")[50:100, 50: 100]
watermark = cv2.imread("watermark1.png")
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
ret, watermark = cv2.threshold(watermark, 64, 255, cv2.THRESH_BINARY);

cv2.namedWindow('image')
cv2.setMouseCallback('image', insert_black_and_white_picture)
while 1:
    cv2.imshow('image', img3)
    if cv2.waitKey(1) & 0xFF == 13:  # Enter
        img3 = img_back3.copy()
        break
cv2.destroyAllWindows()

# 将水印填充成与原图片相同大小
r, c = img3.shape  # 读取原始载体图像的 shape 值
width, height = watermark.shape[:2]
watermark = cv2.copyMakeBorder(watermark, y1, r - y1 - width, x1, c - x1 - width, cv2.BORDER_CONSTANT, value=(255, 255, 255))
print(watermark.shape)
# 将水印图像内的值 255 处理为 1，以方便嵌入
w=watermark[:, :] > 0
watermark[w] = 1

# ========嵌入过程========
t254 = np.ones((r, c), dtype=np.uint8) * 254  # 生成元素值都是 254 的数组
lenaH7 = cv2.bitwise_and(img3, t254)  # 获取 lena 图像的高七位
e = cv2.bitwise_or(lenaH7, watermark)  # 将 watermark 嵌入 lenaH7 内

cv2.imshow('picture', e)
cv2.waitKey()
cv2.destroyAllWindows()

#======提取过程=========
t1 = np.ones((r, c), dtype=np.uint8)
wm = cv2.bitwise_and(e,t1)  # 从载体图像内提取水印图像

#将水印图像内的值 1 处理为 255，以方便显示
w = wm[:, :] > 0
wm[w] = 255
cv2.imshow('wm', wm)
cv2.waitKey()
cv2.destroyAllWindows()
print("练习题4结束")