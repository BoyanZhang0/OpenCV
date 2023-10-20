import cv2
import numpy as np


class Filter():
    """图形滤波与增强部分"""
    BlurType = ['Mean', 'Mediam', 'Gaussian']

    def __init__(self, img=None):
        if img != None:
            self.image = img

    def set_image(self, img):
        if img != None:
            self.image = img

    def blur(self, kernel_size: int, img, method: str = 'Mean', sigma=0):
        """图像平滑处理"""
        dst = None
        if method == 'Mediam':
            dst = cv2.medianBlur(img, (kernel_size, kernel_size))
        elif method == 'Mean':
            dst = cv2.blur(img, (kernel_size, kernel_size))
        else:
            dst = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
        return dst

    def motion_blur(self, kernel_size: int, direction=0, img=None):
        """运动模糊"""
        dst = None
        degree = np.abs(direction) % 180
        channels = img.shape[2]
        # 创建滤波器
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        M = cv2.getRotationMatrix2D((int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)), degree, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()

        # 对每个通道进行滤波处理
        mat = list(cv2.split(img))
        for i in range(channels):
            mat[i] = cv2.filter2D(mat[i], -1, kernel)
        # 组合通道
        dst = np.stack(mat, axis=channels - 1)
        return dst

    def sharpening(self, img, intensity=1.0):
        """
        实现图像锐化
        采用拉普拉斯锐化
        """
        kernel_1 = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])

        kernel_2 = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

        channels = img.shape[2]
        # 处理相关系数
        kernel = kernel_2
        mat = list(cv2.split(img))
        for i in range(channels):
            mat[i] = cv2.filter2D(mat[i], -1, kernel)
        dst = np.stack(mat, axis=2)
        dst = cv2.addWeighted(img, 1 - intensity, dst, intensity, 0)
        return dst

    def edge_detection(self, img, method='Laplacian', high_threshold=50, low_threshold=240):
        """边缘检测算法"""
        channels = img.shape(2)
        mat = list(cv2.split(img))
        for i in range(channels):
            mat[i] = cv2.GaussianBlur(img, ksize=3, sigmaX=0, sigmaY=0)
            if method == 'Laplacian':
                mat[i] = cv2.Laplacian(img, cv2.CV_64F)
            else:
                mat[i] = cv2.Canny(img, low_threshold, high_threshold)
        dst = np.stack(mat, axis=2)
        return dst

    def image_Embossing(self, img, intensity):
        """图像浮雕实现"""
        if intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be in the range [0, 1]")

        # 创建一个浮雕滤波器
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]], dtype=np.float32)

        # 对输入图像应用滤波器
        embossed_image = cv2.filter2D(img, -1, kernel)

        # 调整浮雕效果的强度
        embossed_image = cv2.addWeighted(img, 1 - intensity, embossed_image, intensity, 0)

        return embossed_image

    def vignette(self, radius, img, center: tuple = (-1, -1)):
        """实现图像渐晕"""
        rows, cols, channels = img.shape
        if center == (-1, -1):
            center = (int(rows / 2), int(cols / 2))
            # 产生渐晕的高斯掩膜
        kernel_x = cv2.getGaussianKernel(2 * cols, radius)
        kernel_y = cv2.getGaussianKernel(2 * rows, radius)
        kernel = kernel_y * kernel_x.T
        # 取渐晕中心
        mask_start = (rows - center[0], cols - center[1])
        mask = kernel[mask_start[0]:mask_start[0] + rows, mask_start[1]: mask_start[1] + cols]
        mask = 255 * mask / np.linalg.norm(mask)
        dst = np.copy(img)
        # 应用掩膜
        for i in range(channels):
            dst[:, :, i] = dst[:, :, i] * mask
        return dst

    def contrast_enhancement(self, image, value):
        """图像对比度增强"""
        # value参数的取值范围为[-16, 16]
        value = np.exp(value/50)
        dst = image * float(value)
        dst[dst > 255] = 255
        dst = np.round(dst)
        dst = dst.astype(np.uint8)
        return dst

    def bilateral_filter(self, image, d, sigma_color, sigma_space):
        """实现双边滤波"""
        # 应用双边滤波器
        filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        return filtered_image

    def pixelate_region(self, img, region:tuple , block_size = 10):
        """
        像素化图像的特定区域。

        参数：
        img：输入图像。
        region：一个元组（x，y，宽度，高度），指定要像素化的区域。
        block_size：像素块的大小。

        返回值：
        具有指定区域像素化的图像。
        """
        x, y, width, height = region
        sub_image = img[y:y + height, x:x + width]
        sub_image = cv2.resize(sub_image, (width // block_size, height // block_size))
        sub_image = cv2.resize(sub_image, (width, height), interpolation=cv2.INTER_NEAREST)
        dst = img.copy()
        dst[y:y + height, x:x + width] = sub_image
        return dst

    def image_segmentation(self, input_image, segmentation_color=[255, 0, 0]):
        """
        图像分割，分割的结果支持保存透明图像

        参数：
        input_img:输入图像
        output_image_path:结果输出路径
        segmentation_color = [255, 0, 0]: 分割颜色，默认是红色

        返回值：
        已分割的图像

        """
        if input_image.shape[2] == 4:
            # 如果输入图像包含Alpha通道，则转换为BGR颜色空间
            image = cv2.cvtColor(input_image, cv2.COLOR_BGRA2BGR)
        else:
            image = input_image
        # 创建一个掩码图像，与输入图像大小相同，用于标记分割区域
        mask = np.zeros(image.shape, dtype=np.uint8)

        # 定义一个阈值，用于确定哪些像素属于分割区域
        threshold = 200  # 可根据需要进行调整

        # 使用循环遍历输入图像的每个像素
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                # 计算像素颜色与分割颜色之间的欧氏距离
                distance = np.sqrt(np.sum((image[x, y] - segmentation_color) ** 2))

                # 如果距离小于阈值，则将该像素标记为分割区域
                if distance < threshold:
                    mask[x, y] = image[x, y]
                else:
                    mask[x, y] = [0, 0, 0]

        # 将掩码图像保存为带Alpha通道的PNG格式，以实现透明效果
        # cv2.imwrite(output_image_path, mask)
        return mask

    def grabcut(self, image, rect):
        """图像分割"""
        if rect == (0,0,0,0):
            return image
        bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组
        fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组
        mask = np.zeros(image.shape[:2],np.uint8) # mask遮罩图像
        img = image.copy()
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8') # 0和2做背景
        result = img*mask2[:,:,np.newaxis] # 使用蒙板来获取前景区域
        return result

    def BGR2BGRA(self, image):
        """将bgr图像转换为bgra图像, 并根据阈值设置透明度"""
        # 设置透明度
        lower_bound = np.array([0, 0, 0], dtype=np.uint8)  # 最低阈值（黑色）
        upper_bound = np.array([200, 200, 200], dtype=np.uint8)  # 最高阈值（白色)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        bgra_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bgra_image[:, :, 3] = mask
        return bgra_image




    def inpainting(self, image, ROI):
        """
        图像修复或目标删除。

        参数：
        image：输入图像。
        ROI：感兴趣区域 (Region of Interest)，表示需要修复或删除的区域，格式为 (x, y, width, height)。

        返回值：
        修复或目标删除后的图像。(ROI选取)
        """
        # 将输入图像复制到新变量，以免修改原始图像
        inpainted_image = image.copy()

        # 提取感兴趣区域的坐标和尺寸
        x, y, width, height = ROI

        # 创建一个与感兴趣区域相同大小的掩膜
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        print(ROI)


        # 在掩膜上绘制一个矩形，将感兴趣区域标记为待修复区域 这里创建了一个白色的掩膜，就是删除的作用
        cv2.rectangle(mask, (x, y), (x + width, y + height), (255, 255, 255), thickness=-1)

        # 使用 inpaint 函数来修复或删除目标
        inpainted_image = cv2.inpaint(inpainted_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return inpainted_image

    def create_gif(self, image_list, output_path, duration=100):
        """
        从图像列表创建GIF动画。

        参数：
        image_list：包含在GIF中的图像列表。
        output_path：保存GIF的路径。
        duration：每帧的持续时间（以毫秒为单位）。

        返回值：
        无
        """
        # 初始化VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 1000 / duration, (image_list[0].shape[1], image_list[0].shape[0]))

        # 将每一帧写入视频
        for frame in image_list:
            out.write(frame)

        out.release()

    def change_size(self, image, size:tuple = None , x=1.0, y=1.0):
        # size为resize后的图片大小，形式为（x， y），其中x， y取值为正数即可
        # x， y为比例缩放，范围都为0~1的小数
        dst = cv2.resize(image, dsize=size, fx=x, fy=y, interpolation=cv2.INTER_LINEAR)
        return dst

    def my_flip(self, image, kind):  # type为翻转类型，范围为-1，0， 1，使用滚动条选择
        # 图像翻转
        dst = cv2.flip(image, kind)
        return dst

    def affine(self, image, mat_src_param = None, mat_dst_param = None):  # 可添加传入参数mat_src, mat_dst
        # 仿射变换
        height, width = image.shape[:2]
        # 此处使用的mat_src， mat_dst为默认值，如果可以读取鼠标点击的三个点，可以选择并不使用默认array，而改为将三个点作为参数传入函数中
        if mat_src_param is None:
            mat_src = np.float32([[0, 0], [0, height], [width, 0]])
        else:
            # 规范化mat_src, 按顺序获取三个点, 左上,右上,左下
            sums = mat_src_param.sum(axis=1)
            # 找到左上角点的索引
            left_top_index = np.argmin(sums)
            # 找到左下角点的索引
            left_bottom_index = np.argmax(mat_src_param[:, 1])
            # 找到右上角点的索引
            right_top_index = np.argmax(mat_src_param[:, 0])
            # 重新排列点的顺序，以左上、左下、右上的顺序
            mat_src = np.array([mat_src_param[left_top_index],
                                    mat_src_param[left_bottom_index],
                                    mat_src_param[right_top_index]])

        if mat_dst_param is None:
            mat_dst = np.float32([[0, 0], [100, height - 100], [width - 100, 100]])
        else:
            # 规范化mat_dst, 按顺序获取三个点, 左上,右上,左下
            sums = mat_dst_param.sum(axis=1)
            # 找到左上角点的索引
            left_top_index = np.argmin(sums)
            # 找到左下角点的索引
            left_bottom_index = np.argmax(mat_dst_param[:, 1])
            # 找到右上角点的索引
            right_top_index = np.argmax(mat_dst_param[:, 0])
            # 重新排列点的顺序，以左上、左下、右上的顺序
            mat_dst = np.array([mat_dst_param[left_top_index],
                                    mat_dst_param[left_bottom_index],
                                    mat_dst_param[right_top_index]])

        tran = cv2.getAffineTransform(mat_src, mat_dst)
        dst = cv2.warpAffine(image, tran, (width, height))
        return dst

    def perspective(self, image, mat_src, mat_dst):  # 可添加传入参数mat_src, mat_dst
        # 透视变换
        height, width = image.shape[:2]
        # mat_src， mat_dst同仿射变换，仅由3个点变为4个点
        mat_src = np.float32([[0, 0], [0, width], [height, 0], [height, width]])
        mat_dst = np.float32([[50, 100], [height - 50, 100], [50, width - 100], [height - 50, width - 100]])
        tran = cv2.getPerspectiveTransform(mat_src, mat_dst)
        dst = cv2.warpPerspective(image, tran, (width, height))
        return dst

    # def my_concave(self):
    #     # 凹透镜滤镜
    #     height, width = self.image.shape[:2]
    #     center = (width // 2, height // 2)
    #     dst = self.image.copy()
    #     for y in range(height):
    #         for x in range(width):
    #             theta = np.arctan2(y - center[1], x - center[0])
    #             r2 = int(np.sqrt(np.linalg.norm(np.array([x, y]) - np.array(center))) * 8)
    #             newX = center[0] + int(r2 * np.cos(theta))
    #             newY = center[1] + int(r2 * np.sin(theta))
    #             if newX < 0:
    #                 newX = 0
    #             elif newX >= width:
    #                 newX = width - 1
    #             if newY < 0:
    #                 newY = 0
    #             elif newY >= height:
    #                 newY = height - 1
    #             dst[y, x] = self.image[newY, newX]
    #     return dst

    def my_convex(self, img):
        # 凸透镜滤镜
        height, width, channel = img.shape

        dst = np.zeros([height, width, channel], dtype=np.uint8)
        center_x = height / 2
        center_y = width / 2

        radius = min(center_x, center_y)
        for i in range(height):
            for j in range(width):
                distance = ((i - center_x) * (i - center_x) + (j - center_y) * (j - center_y))
                new_dist = np.sqrt(distance)
                dst[i, j, :] = img[i, j, :]
                if distance <= radius ** 2:
                    new_i = int(np.floor(new_dist * (i - center_x) / radius + center_x))
                    new_j = int(np.floor(new_dist * (j - center_y) / radius + center_y))
                    dst[i, j, :] = img[new_i, new_j, :]
        return dst
    

    def my_outline(self, img):
        # 获取图片轮廓
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化到灰度空间
        ret, binary = cv2.threshold(img_gray, 120, 220, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 得到图像轮廓
        zero_mask = np.zeros(img_gray.shape, np.uint8)
        dst = cv2.drawContours(zero_mask, contours, -1, (255, 255, 255), -1)  # 绘制图像轮廓
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        print(dst.shape)
        return dst

        
    def my_concave(self, img):
            # 凹透镜滤镜
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            dst = img.copy()
            for y in range(height):
                for x in range(width):
                    theta = np.arctan2(y - center[1], x - center[0])
                    r2 = int(np.sqrt(np.linalg.norm(np.array([x, y]) - np.array(center))) * 8)
                    newX = center[0] + int(r2 * np.cos(theta))
                    newY = center[1] + int(r2 * np.sin(theta))
                    if newX < 0:
                        newX = 0
                    elif newX >= width:
                        newX = width - 1
                    if newY < 0:
                        newY = 0
                    elif newY >= height:
                        newY = height - 1
                    dst[y, x] = img[newY, newX]
            return dst



    def insert_word(self, image, point, text='image', size=3, r=255, g=255, b=255):
        # point为鼠标点击位置，形式为（x， y）
        # size为插入的字体大小，范围为1~5， r，g，b为颜色，范围为0~255。都使用滚动条进行选择
        # 插入文字
        dst = image.copy()
        cv2.putText(dst, text, point, cv2.FONT_HERSHEY_SIMPLEX, size, (r, g, b), 5)
        return dst

    def insert_img(self, image:np.ndarray, point, pict=None):  # point为鼠标点击位置，形式为（x， y）
        # 插入图像
        x, y = point
        height, width = pict.shape[:2]
        dst = image.copy()
        rows, cols = image.shape[:2]
        if y + width > rows:
            width = rows - y
        if x + height > cols:
            height = cols - x
        dst[y: y + width, x: x + height] = pict[:width, :height]
        return dst

    def change_bright(self, image, value):  # value为正时表示增加亮度，为负时表示降低亮度
        dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(dst)
        v1 = np.clip(cv2.add(1 * v, value), 0, 255)
        dst = cv2.merge((h, s, v1))
        dst = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)
        return dst


    def Clip(self, image:np.ndarray, ROI:tuple):
        """根据选定的ROI裁剪图片"""
        rect = ROI
        dst = np.zeros_like(image)
        if rect[0] >= 0 and rect[1] >=0 and rect[2] > 0 and rect[3] > 0:
            dst = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return dst

