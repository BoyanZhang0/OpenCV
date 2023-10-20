import sys
from typing import Union
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget
import cv2
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qframelesswindow import *
from qfluentwidgets import *

from .UI_image import Widget_Image
from .Core import Filter


class Widget_segmentation(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setObjectName('UI_segmentation')

    def init_ui(self):
        #创建算法实例
        self.filter = Filter()
        self.container = QVBoxLayout(self)
        # 创建备份图片, 仅在读取图片和保存修改时修改内容, 在重置修改时调用
        self.image_backup = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # 白色背景
        # 加载显示图片类型
        self.widgetImage = Widget_Image(self.image_backup)

        self.btn_save_change = PushButton('保存修改')
        self.btn_reset = PushButton('重置修改')
        # Todo
        self.btn_segmentation = PushButton('图像分割')
        self.btn_saveBGRA = PushButton('转换为透明底png图像')
        self.layout_outer = QHBoxLayout()
        self.layout_outer.addWidget(self.btn_segmentation)
        self.layout_outer.addWidget(self.btn_saveBGRA)
        self.layout_outer.addWidget(self.btn_save_change)
        self.layout_outer.addWidget(self.btn_reset)
        # Todo
        self.container.addWidget(self.widgetImage)
        self.container.addLayout(self.layout_outer)
        # 设置布局
        self.setLayout(self.container)
        # 绑定信号槽
        self.BindSlot()

    def BindSlot(self):
        #Todo
        self.btn_reset.clicked.connect(self.Reset_image)
        self.btn_save_change.clicked.connect(self.Save_change_image)
        self.btn_segmentation.clicked.connect(self.grabcut)
        self.btn_saveBGRA.clicked.connect(self.SaveBGRA)
        pass

    def paintEvent(self, a0: QPaintEvent):
        super().paintEvent(a0)
        self.widgetImage.update()


    def Set_image(self, param: Union[str, np.ndarray]):
        """仅用于加载, 请勿用于更新修改图片内容, 更新请使用widget_image对象中的Set_image"""
        self.widgetImage.Set_image(param)
        self.image_backup = self.Get_image()
        self.update()

    def Get_image(self):
        """返回图像"""
        return self.widgetImage.Get_image()

    def Reset_image(self):
        """用于重载图片, 重置修改"""
        self.widgetImage.Set_image(self.image_backup)
        self.update()

    def Save_change_image(self):
        """保存修改"""
        self.image_backup = self.Get_image()

    # def Segmentation(self):
    #     """图像分割"""
    #     self.widgetImage.Set_image(self.filter.image_segmentation(self.Get_image()))
    #     self.update()

    def SaveBGRA(self):
        """转换为透明底png图像"""
        self.widgetImage.Set_image(self.filter.BGR2BGRA(self.Get_image()))
        self.update()

    def grabcut(self):
        self.widgetImage.Set_image(self.filter.grabcut(self.Get_image(), self.widgetImage.Get_ROI()))
        self.update()