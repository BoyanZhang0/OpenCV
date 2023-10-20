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

class Widget_filter(QFrame):
    """测试, 用于展示图片功能"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # 创建滤波器对象
        self.filter = Filter()
        #初始化UI界面
        self.init_ui()
        self.setObjectName("UI_filter")

    def init_ui(self):
        # 创建布局
        self.container = QVBoxLayout(self)
        #创建备份图片, 仅在读取图片和保存修改时修改内容, 在重置修改时调用
        self.image_backup = np.ones((1000,1000,3), dtype=np.uint8)*255  # 白色背景
        # 加载显示图片类型
        self.widgetImage = Widget_Image(self.image_backup)
        # 创建按钮
        self.layout_button = QHBoxLayout()
        self.bt_meanblur = PushButton('均值滤波')
        self.bt_motionblur = PushButton('运动模糊')
        self.bt_sharpening = PushButton('锐化')
        self.bt_savechange = PushButton('保存修改')
        self.bt_reset = PushButton('重置修改')
        # 添加按钮
        self.layout_button.addWidget(self.bt_meanblur)
        self.layout_button.addWidget(self.bt_motionblur)
        self.layout_button.addWidget(self.bt_sharpening)
        self.layout_button.addWidget(self.bt_savechange)
        self.layout_button.addWidget(self.bt_reset)
        # 绑定信号槽
        self.BindSlot()
        # 添加控件
        self.container.addWidget(self.widgetImage)
        self.container.addLayout(self.layout_button)
        self.setLayout(self.container)

    def BindSlot(self):
        self.bt_meanblur.clicked.connect(self.Blur)
        self.bt_motionblur.clicked.connect(self.motion_blur)
        self.bt_sharpening.clicked.connect(self.sharpening)
        self.bt_reset.clicked.connect(self.Reset_image)
        self.bt_savechange.clicked.connect(self.Save_change_image)

    def Blur(self):
        """均值滤波"""
        self.widgetImage.Set_image(self.filter.blur(5, self.Get_image()))
        self.update()

    def motion_blur(self):
        """运动模糊"""
        self.widgetImage.Set_image(self.filter.motion_blur(5, 45, self.Get_image()))
        self.update()

    def sharpening(self):
        """锐化"""
        self.widgetImage.Set_image(self.filter.sharpening(self.Get_image()))
        self.update()

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
        self.widgetImage.Set_image(self.image_backup)
        self.update()

    def Save_change_image(self):
        self.image_backup = self.Get_image()
