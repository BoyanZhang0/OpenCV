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


class Widget_effects(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setObjectName('UI_effect')

    def init_ui(self):
        #创建算法实例
        self.filter = Filter()
        self.container = QVBoxLayout(self)
        
        # 创建备份图片, 仅在读取图片和保存修改时修改内容, 在重置修改时调用
        self.image_backup = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # 白色背景
        
        # 加载显示图片类型
        self.widgetImage = Widget_Image(self.image_backup)
        
        #创建布局
        self.layout_h_layout = QHBoxLayout()
        self.layout_saveandreset = QVBoxLayout()
        # 创建按钮
        self.slider_mosaic_label = QLabel('马赛克块大小: 10')
        self.bt_mosaic = PushButton('打码')
        self.slider_mosaic = Slider(Qt.Horizontal, self)
        self.slider_mosaic.setMinimum(1)
        self.slider_mosaic.setMaximum(100)
        self.slider_mosaic.setValue(10)
        self.slider_mosaic.resize(100, 30)
        self.mosaic_layout = QVBoxLayout()
        self.mosaic_layout.addWidget(self.slider_mosaic_label)
        self.mosaic_layout.addWidget(self.slider_mosaic)
        self.mosaic_layout.addWidget(self.bt_mosaic)


        self.bt_concave = PushButton('凹透镜特效')
        self.bt_convex = PushButton('凸透镜特效')
        self.bt_outline = PushButton('生成轮廓')
        self.bt_savechange = PushButton('保存修改')
        self.bt_reset = PushButton('重置修改')
        self.layout_saveandreset.addWidget(self.bt_savechange)
        self.layout_saveandreset.addWidget(self.bt_reset)
        #添加按钮  
        self.layout_h_layout.addWidget(self.bt_concave)
        self.layout_h_layout.addWidget(self.bt_convex)
        self.layout_h_layout.addWidget(self.bt_outline)
        self.layout_h_layout.addLayout(self.mosaic_layout)
        self.layout_h_layout.addLayout(self.layout_saveandreset)

        # 设置布局
        self.container.addWidget(self.widgetImage)
        self.container.addLayout(self.layout_h_layout)
        self.setLayout(self.container)
        # 绑定信号槽
        self.BindSlot()

    def BindSlot(self):
        #Todo
        self.bt_concave.clicked.connect(self.Concave)
        self.bt_convex.clicked.connect(self.Convex)
        self.bt_outline.clicked.connect(self.Outline)
        self.bt_reset.clicked.connect(self.Reset_image)
        self.bt_savechange.clicked.connect(self.Save_change_image)
        self.bt_mosaic.clicked.connect(self.mosaic)
        self.slider_mosaic.valueChanged.connect(self.slider_mosaic_change)
        pass

    def Concave(self):
        self.widgetImage.Set_image(self.filter.my_concave(self.Get_image()))
        self.update()
        pass

    def Convex(self):
        self.widgetImage.Set_image(self.filter.my_convex(self.Get_image()))
        self.update()
        pass

    def Outline(self):
        self.widgetImage.Set_image(self.filter.my_outline(self.Get_image()))
        self.update()
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

    def mosaic(self):
        """马赛克"""
        self.widgetImage.Set_image(self.filter.pixelate_region(self.Get_image(),self.widgetImage.Get_ROI(), self.slider_mosaic.value()))
        self.update()
    
    def slider_mosaic_change(self):
        self.slider_mosaic_label.setText('马赛克块大小: '+str(self.slider_mosaic.value()))
        self.update()