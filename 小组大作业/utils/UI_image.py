from typing import Union

import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qframelesswindow import *

from qfluentwidgets import *


class Wideget_ImageLabel(PixmapLabel):
    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.init_param(image)
        # self.setMinimumSize(800,500)

    def init_param(self, image: np.ndarray):
        # 图片相关参数
        self.image_cv = image
        self.image_qt = self.image_convert_cv2qt(self.image_cv)
        self.pixmap = QPixmap()
        self.scaled_pixmap = QPixmap()
        self.raw_size = image.shape[:2]
        # 选取(点击,画框相关参数)
        self.click_pos = (-1, -1)
        self.click_posPost = (-1, -1)
        self.click_rectROI = (-1, -1, -1, -1)
        # 选取, 实际处理区域
        self.pos = (-1, -1)
        self.pos_post = (-1, -1)
        self.rectROI = (-1, -1, -1, -1)
        # 缩放比例
        self.scale_factor = 1.0
        # 控制选取参数
        self.click = False
        # 控制绘制相关的参数
        self.opacity = 0.5
        self.radius = 10
        self.rectwidth = 2
        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.reduce_opacity)

    def Get_ROI(self):
        # 返回ROI区域
        # ROI有效, 值都大于等于0
        if self.rectROI[0] >= 0 and self.rectROI[1] >= 0 and self.rectROI[2] > 0 and self.rectROI[3] > 0:
            return self.rectROI
        else:
            return (0,0,0,0)

    def GetPoint(self):
        # 返回点击点
        # 点击点有效, 值都大于等于0
        if self.pos[0] >= 0 and self.pos[1] >= 0:
            return self.pos
        else:
            print(self.pos)
            return (-1,-1)

    def resetImage(self, image: np.ndarray):
        """用于重置图片"""
        self.image_cv = image
        self.update_label()
        self.update()

    def update_label(self):
        """用于图片修改后更新图片, 此处不进行update, 由外部调用"""
        self.image_qt = self.image_convert_cv2qt(self.image_cv)
        self.pixmap = QPixmap.fromImage(self.image_qt)
        self.scale_image()
        self.update()

    def reduce_opacity(self):
        if self.opacity > 0:
            self.opacity -= 0.5
            self.update()
        else:
            self.opacity = 0
            self.timer.stop()
            self.click = False
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if event.buttons() == Qt.LeftButton:
            self.opacity = 0.5
            self.click_pos = (event.localPos().x(), event.localPos().y())
            self.click_posPost = self.click_pos
            # 计算实际图像的像素位置
            if self.click_pos[0] >=0 and self.click_pos[1] >=0:
                self.pos = (int(self.click_pos[0] / self.scale_factor), int(self.click_pos[1] / self.scale_factor))
            self.click_rectROI = (-1, -1, -1, -1)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        self.click_posPost = (event.pos().x(), event.pos().y())
        self.click_rectROI = (
            min(self.click_pos[0], self.click_posPost[0]), min(self.click_pos[1], self.click_posPost[1]),
            qAbs(self.click_pos[0] - self.click_posPost[0]), qAbs(self.click_pos[1] - self.click_posPost[1]))
        self.pos_post = (int(self.click_posPost[0] / self.scale_factor), int(self.click_posPost[1] / self.scale_factor))
        self.rectROI = (min(self.pos[0], self.pos_post[0]), min(self.pos[1], self.pos_post[1]),
                        int(qAbs(self.pos[0] - self.pos_post[0])), int(qAbs(self.pos[1] - self.pos_post[1])))
        self.releaseMouse()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.timer.start(100)
            if self.click_pos == self.click_posPost:
                # 代表点击
                self.opacity = 1.0
                self.click = True
        self.releaseMouse()

    def paintEvent(self, event):
        super().paintEvent(event)
        # 绘制选取框
        if self.click == False:
            painter = QPainter(self)
            # 添加抗锯齿
            painter.setRenderHints(QPainter.Antialiasing)
            # 设置画笔无边框
            pen = QPen(Qt.NoPen)
            painter.setPen(pen)
            # 设置画刷,浅蓝色
            brush = QBrush(QColor(137, 207, 240, int(self.opacity * 255)))
            painter.setBrush(brush)
            # 绘制点击图案
            if self.click_rectROI[0] >= 0 and self.click_rectROI[1] >= 0 and self.click_rectROI[2] >= 0 and \
                    self.click_rectROI[3] >= 0:
                painter.drawRect(self.click_rectROI[0], self.click_rectROI[1], self.click_rectROI[2],
                                 self.click_rectROI[3])

        # 绘制点击的点型区域
        if self.click == True:
            painter = QPainter(self)
            # 添加抗锯齿
            painter.setRenderHints(QPainter.Antialiasing)
            # 设置画笔无边框
            pen = QPen(Qt.NoPen)
            painter.setPen(pen)
            # 设置画刷, 实现逐渐消失功能
            brush = QBrush(QColor(0, 0, 0, int(self.opacity * 255)))
            painter.setBrush(brush)
            # 绘制点击图案
            if self.click_pos[0] >= 0 and self.click_pos[1] >= 0:
                painter.drawEllipse(int(self.click_pos[0] - self.radius / 2), int(self.click_pos[1] - self.radius / 2),
                                    self.radius, self.radius)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            # 滚轮向上滚动，放大图像
            self.scale_factor += 0.05
            # 限制放大比例
            if self.scale_factor > 10.0:
                self.scale_factor = 10.0
        elif event.angleDelta().y() < 0:
            # 滚轮向下滚动，缩小图像
            self.scale_factor -= 0.05
            # 缩放比例不能小于0.05
            if self.scale_factor < 0.05:
                self.scale_factor = 0.05
        self.scale_image()

    def scale_image(self):
        # 根据缩放比例对图片进行缩放
        self.scaled_pixmap = self.pixmap.scaled(int(self.raw_size[1] * self.scale_factor),
                                                int(self.raw_size[0] * self.scale_factor),
                                                Qt.KeepAspectRatio)
        self.setPixmap(self.scaled_pixmap)
        self.update()

    def image_convert_cv2qt(self, image_cv):
        """将cv2的图片转换为qt的QImage格式"""
        #判断通道数, 如果为4通道, 则去掉alpha通道
        # print(image_cv.shape)
        if image_cv.shape[2] == 4:
            temp = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGB)
        elif image_cv.shape[2] == 3:
            temp = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        img = QImage(temp.data, temp.shape[1], temp.shape[0], temp.shape[1]*3, QImage.Format_RGB888)
        return img

    def image_convert_qt2cv(self, image_qt):
        """将qt的QImage格式转换为cv2的图片"""
        img = image_qt.constBits()
        img = img.setsize(image_qt.byteCount())
        img = np.array(img).reshape(image_qt.height(), image_qt.width(), 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def setImage_frompath(self, path: str):
        """加载图片"""
        self.image_cv = cv2.imread(path)
        #获取图片原始大小
        self.raw_size = self.image_cv.shape[:2]
        # 重置缩放比例
        self.scale_factor = 1.0
        # 重置选取区域
        self.click_pos = (-1, -1)
        self.click_posPost = (-1, -1)
        self.click_rectROI = (-1, -1, -1, -1)
        self.pos = (-1, -1)
        self.pos_post = (-1, -1)
        self.rectROI = (-1, -1, -1, -1)
        # 重置显示相关的内容
        self.image_qt = self.image_convert_cv2qt(self.image_cv)
        self.pixmap = QPixmap.fromImage(self.image_qt)
        self.scale_image()
        self.update()

    def setImage_fromcv(self, image_cv):
        """加载图片,也可用于更新图片"""
        self.image_cv = image_cv
        # 获取图片原始大小
        self.raw_size = self.image_cv.shape[:2]
        # 重置缩放比例
        self.scale_factor = 1.0
        # 重置选取区域
        self.click_pos = (-1, -1)
        self.click_posPost = (-1, -1)
        self.click_rectROI = (-1, -1, -1, -1)
        self.pos = (-1, -1)
        self.pos_post = (-1, -1)
        self.rectROI = (-1, -1, -1, -1)
        self.click = False
        # 重置显示相关的内容
        self.image_qt = self.image_convert_cv2qt(self.image_cv)
        self.pixmap = QPixmap.fromImage(self.image_qt)
        self.scale_image()
        self.update()

class Widget_Image(QWidget):
    def __init__(self, image:np.ndarray = None, parent=None):
        super().__init__(parent)
        self.init_ui(image)

    def init_ui(self, image:np.ndarray):
        self.image_label = Wideget_ImageLabel(image,self)
        # 创建并设置滚动条
        self.scroll_area = SmoothScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.scroll_area.setMinimumSize(800, 500)
        # 创建布局器
        self.container = QHBoxLayout(self)
        self.container.addWidget(self.scroll_area)
        self.setLayout(self.container)
        # self.resize(500, 400)

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        self.image_label.update()
        self.scroll_area.update()

    def eventFilter(self, widget: QObject, event: QEvent) -> bool:
        if widget == self and event.type() == QEvent.Wheel:
            self.image_label.event(event)
            return True
        return super().eventFilter(widget, event)

    def Get_ROI(self):
        return self.image_label.Get_ROI()

    def Get_point(self):
        return self.image_label.GetPoint()

    def Get_image(self):
        """获取当前图片"""
        return self.image_label.image_cv

    def Set_image(self, param: Union[None, str, np.ndarray]):
        """加载, 更新, 修改图片内容"""
        if param is None:
            return None
        elif type(param) == str:
            self.image_label.setImage_frompath(param)
        elif type(param) == np.ndarray:
            self.image_label.setImage_fromcv(param)
        self.update()

# # 运行示例
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     w = Widget_Image()
#     # w.load_image('../resources/figure/img1.jpg')
#     w.show()
#     app.exec()
