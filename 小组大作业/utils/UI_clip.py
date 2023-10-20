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

class Widget_clip(QFrame):
    def __init__(self, parent = None):
        super().__init__(parent = parent)
        self.init_ui()
        self.setObjectName('UI_clip')

    def init_ui(self):
        #创建算法实例
        self.filter = Filter()
        self.container = QVBoxLayout(self)
        # 创建备份图片, 仅在读取图片和保存修改时修改内容, 在重置修改时调用
        self.image_backup = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # 白色背景
        # 加载显示图片类型
        self.widgetImage = Widget_Image(self.image_backup)
        # 创建裁剪按钮
        self.btn_clip = PushButton("裁剪图片")
        self.btn_save = PushButton("保存修改")
        self.btn_reset = PushButton("重置修改")
        self.btn_resize = PushButton("图像缩放")
        self.line_width = LineEdit()
        self.line_width.setPlaceholderText("宽度")
        self.line_height = LineEdit()
        self.line_height.setPlaceholderText("高度")
        self.line_width_scale = LineEdit()
        self.line_width_scale.setPlaceholderText("宽度缩放比例")
        self.line_height_scale = LineEdit()
        self.line_height_scale.setPlaceholderText("高度缩放比例")
        self.layout_resize_line = QHBoxLayout()
        self.layout_resize_line.addWidget(self.line_width)
        self.layout_resize_line.addWidget(self.line_height)
        self.layout_resize_line.addWidget(self.line_width_scale)
        self.layout_resize_line.addWidget(self.line_height_scale)
        self.layout_resize = QVBoxLayout()
        self.layout_resize.addLayout(self.layout_resize_line)
        self.layout_resize.addWidget(self.btn_resize)
        # 创建图像翻转Combox, 以及确认按钮
        self.combox_flip = ComboBox()
        # self.combox_flip.setText("图像翻转")
        self.combox_flip.addItem("水平翻转")
        self.combox_flip.addItem("垂直翻转")
        self.combox_flip.addItem("水平垂直翻转")
        self.combox_flip.setCurrentIndex(0)
        self.btn_flip = PushButton("确认翻转")
        # 创建布局
        self.flip_layout = QHBoxLayout()
        self.flip_layout.addWidget(self.combox_flip)
        self.flip_layout.addWidget(self.btn_flip)
        # 创建底层外部布局器
        self.hlayout_1 = QHBoxLayout(self)
        self.hlayout_1.addStretch(0)
        self.hlayout_1.addLayout(self.layout_resize)
        self.hlayout_1.addWidget(self.btn_clip)
        self.hlayout_1.addLayout(self.flip_layout)
        self.hlayout_1.addWidget(self.btn_save)
        self.hlayout_1.addWidget(self.btn_reset)
        self.hlayout_1.addStretch(0)
        # 添加到容器中
        self.container.addWidget(self.widgetImage)
        self.container.addLayout(self.hlayout_1)
        # 设置布局
        self.setLayout(self.container)
        # 绑定信号槽
        self.BindSlot()


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
        self.line_height.clear()
        self.line_width.clear()
        self.line_height_scale.clear()
        self.line_width_scale.clear()
        self.update()

    def Save_change_image(self):
        """保存修改"""
        self.image_backup = self.Get_image()


    def BindSlot(self):
        self.btn_save.clicked.connect(self.Save_change_image)
        self.btn_reset.clicked.connect(self.Reset_image)

        self.btn_flip.clicked.connect(self.Flip_image)
        self.btn_clip.clicked.connect(self.Clip_image)
        self.btn_resize.clicked.connect(self.Resize_image)



    def Clip_image(self):
        """根据选定的ROI裁剪图片"""
        rect = self.widgetImage.Get_ROI()
        print(rect)
        self.widgetImage.Set_image(self.filter.Clip(self.Get_image(), rect))
        self.update()

    def Flip_image(self):
        """翻转图片"""
        kind = self.combox_flip.currentText()
        val = 0
        if kind == '水平翻转':
            val = 1
        elif kind == '垂直翻转':
            val = 0
        elif kind == '水平垂直翻转':
            val = -1
        self.widgetImage.Set_image(self.filter.my_flip(self.Get_image(), kind=val))
        self.update()

    def Resize_image(self):
        width = self.line_width.text()
        height = self.line_height.text()
        fx = self.line_width_scale.text()
        fy = self.line_height_scale.text()
        if width != '' and height != '':
            width = int(width)
            height = int(height)
            self.widgetImage.Set_image(self.filter.change_size(self.Get_image(), (width, height)))
        elif fx != '' and fy != '':
            fx = float(fx)
            fy = float(fy)
            self.widgetImage.Set_image(self.filter.change_size(self.Get_image(), size=None, x=fx, y=fy))
        else:
            return
        self.update()