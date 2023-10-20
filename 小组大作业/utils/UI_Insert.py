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


class Widget_Insert(QFrame):
    """图像裁剪(resize) """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setObjectName('UI_insert')

    def init_ui(self):
        #创建算法实例
        self.filter = Filter()
        self.container = QVBoxLayout(self)
        # 创建备份图片, 仅在读取图片和保存修改时修改内容, 在重置修改时调用
        self.image_backup = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # 白色背景
        # 加载显示图片类型
        self.widgetImage = Widget_Image(self.image_backup)
        # Todo
        # 创建插入文字按钮和输入文本框
        self.btn_insertTxt = PushButton("插入文字")
        self.line_insertTxt = LineEdit()
        self.line_insertTxt.setPlaceholderText("请输入要插入的文字")
        # 创建布局器
        self.layout_insertTxt = QVBoxLayout()
        self.layout_insertTxt.addWidget(self.line_insertTxt)
        self.layout_insertTxt.addWidget(self.btn_insertTxt)
        # 创建插入图像按钮
        self.btn_insertImage = PushButton("选择图像并插入")
        # Todo
        # 创建重置和保存修改
        self.btn_save_change = PushButton('保存修改')
        self.btn_reset = PushButton('重置修改')
        #创建底层外部布局器
        self.layout_outer = QHBoxLayout()
        self.layout_outer.addLayout(self.layout_insertTxt)
        self.layout_outer.addWidget(self.btn_insertImage)
        self.layout_outer.addWidget(self.btn_save_change)
        self.layout_outer.addWidget(self.btn_reset)
        #添加到总布局中
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
        self.btn_insertTxt.clicked.connect(self.Insert_txt)
        self.btn_insertImage.clicked.connect(self.Insert_image)


    def Insert_txt(self):
        """插入文字"""
        txt = self.line_insertTxt.text()
        print(txt)
        print(self.widgetImage.Get_point())
        if txt == '' or txt is None:
            return
        dst = self.filter.insert_word( image=self.Get_image(), point=self.widgetImage.Get_point(), text=txt, size=3)

        self.widgetImage.Set_image(dst)
        self.update()


    def Insert_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "打开文件", "", "图像文件 (*.jpg *.png);;所有文件 (*)",
                                                   options=options)
        image = cv2.imread(file_name)
        if image is None:
            return
        dst = self.filter.insert_img(image=self.Get_image(), point=self.widgetImage.Get_point(), pict=image)
        self.widgetImage.Set_image(dst)
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
        """用于重载图片, 重置修改"""
        self.widgetImage.Set_image(self.image_backup)
        self.update()

    def Save_change_image(self):
        """保存修改"""
        self.image_backup = self.Get_image()
