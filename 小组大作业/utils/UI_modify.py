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


class Widget_modify(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setObjectName('UI_modify')

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
        # 创建相关控制的变量
        self.control_bright = False
        self.control_contrast = False
        # 创建相关布局和按钮
        self.layout_changeandsave = QVBoxLayout()
        self.layout_changeandsave.addWidget(self.btn_save_change)
        self.layout_changeandsave.addWidget(self.btn_reset)
        self.slider = Slider(Qt.Horizontal,self)
        # 设置滑动条默认值
        self.slider_label = QLabel('调整对比度或亮度', alignment=Qt.AlignCenter)
        self.slider_displayvalue = QLabel('0', alignment=Qt.AlignCenter)
        self.slider.setValue(0)
        self.slider.setRange(-100, 100)
        self.slider.setSingleStep(1)
        self.slider.resize(200, 30)
        self.slider_layout = QVBoxLayout()
        self.slider_layout.addWidget(self.slider_label)
        self.slider_layout.addWidget(self.slider_displayvalue)
        self.slider_layout.addWidget(self.slider)
        self.btn_contrast = PushButton('调整对比度')
        self.btn_brightness = PushButton('调整亮度')
        self.layout_btn_modify = QVBoxLayout()
        self.layout_btn_modify.addWidget(self.btn_contrast)
        self.layout_btn_modify.addWidget(self.btn_brightness)
        self.layout_outer = QHBoxLayout()
        self.layout_outer.addLayout(self.layout_btn_modify)
        self.layout_outer.addLayout(self.slider_layout)
        self.layout_outer.addLayout(self.layout_changeandsave)
        self.container.addWidget(self.widgetImage)
        self.container.addLayout(self.layout_outer)
        # Todo
        # 设置布局
        self.setLayout(self.container)
        # 绑定信号槽
        self.BindSlot()

    def BindSlot(self):
        #Todo
        self.btn_reset.clicked.connect(self.Reset_image)
        self.btn_save_change.clicked.connect(self.Save_change_image)
        self.btn_brightness.clicked.connect(self.Brightness)
        self.btn_contrast.clicked.connect(self.Contrast)
        self.slider.valueChanged.connect(self.change)

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
        self.slider.setValue(0)
        self.update()

    def Save_change_image(self):
        """保存修改"""
        self.image_backup = self.Get_image()

    def Contrast(self):
        self.control_contrast = True
        self.control_bright = False
        self.slider_label.setText('当前调整对比度')
        self.slider.setValue(0)
        self.slider.setRange(-200, 200)
        self.slider.setSingleStep(1)

        self.update()
        pass

    def Brightness(self):
        self.control_bright = True
        self.control_contrast = False
        self.slider_label.setText('当前调整亮度')
        self.slider.setRange(-200, 200)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        pass

    def change(self):
        if self.control_bright:
            self.widgetImage.Set_image(self.image_backup)
            self.widgetImage.Set_image(self.filter.change_bright(self.Get_image(),self.slider.value()))
            self.slider_displayvalue.setText(str(self.slider.value()))
        elif self.control_contrast:
            self.widgetImage.Set_image(self.image_backup)
            self.widgetImage.Set_image(self.filter.contrast_enhancement(self.Get_image(),self.slider.value()))
            self.slider_displayvalue.setText(str(self.slider.value()))
        else:
            self.slider.setValue(0)
        self.update()
