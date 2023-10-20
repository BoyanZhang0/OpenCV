import sys
from typing import Union
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget
import cv2
import numpy as np
import imageio

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qframelesswindow import *
from qfluentwidgets import *

from .UI_image import Widget_Image
from .Core import Filter
from .UI_displayGIF import display_GIF

class Widget_gif(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setObjectName('UI_GIF')
        # self.savefile = display_GIF('resources/cash/tmp.gif',self)
        # self.savefile.hide()

    def init_ui(self):
        #创建算法实例
        self.filter = Filter()
        self.container = QVBoxLayout(self)
        # 创建备份图片, 仅在读取图片和保存修改时修改内容, 在重置修改时调用
        self.gif_movie = QMovie()
        self.gif_label = QLabel()
        self.gif_label.setMovie(self.gif_movie)
        # 加载显示图片类型
        self.btn_select_image = PushButton('选择多个图片')
        self.btn_save_gif = PushButton('另存为')
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.btn_select_image)
        self.btn_layout.addWidget(self.btn_save_gif)
        self.container.addWidget(self.gif_label)
        self.container.addLayout(self.btn_layout)
        self.setLayout(self.container)
        # 绑定信号槽
        self.BindSlot()

    def BindSlot(self):
        #Todo
        self.btn_select_image.clicked.connect(self.CreateTmpGIF)
        self.btn_save_gif.clicked.connect(self.Save_gif)

    def paintEvent(self, a0: QPaintEvent):
        super().paintEvent(a0)

    def CreateTmpGIF(self):
        """选择图片"""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self, 'Select Files', '', 'All Files (*)', options=options)
        images = []
        for file_path in file_paths:
            image = cv2.resize(cv2.imread(file_path), (1000,1000))
            b,g,r = cv2.split(image)
            images.append(cv2.merge([r,g,b]))
        if len(images) == 0:
            return
        imageio.mimsave('resources/cash/tmp.gif', images ,'GIF',duration = 4,loop=0)
        self.gif_movie.setFileName('resources/cash/tmp.gif')
        self.gif_movie.start()
        available_space = (self.gif_label.contentsRect().width(), self.gif_label.contentsRect().height())
        size = min(available_space)
        self.gif_movie.setScaledSize(QSize(size, size))
        # self.gif_label.setScaledContents(True)
        self.update()

    def Set_image(self, param: Union[str, np.ndarray]):
        pass

    def Save_gif(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "另存为", "", "图像文件 (*.gif);;所有文件 (*)",
                                                   options=options)
        if file_name == '':
            return
        src_file = QFile('resources/cash/tmp.gif')
        dst_file = QFile(file_name)
        if not src_file.open(QIODevice.ReadOnly):
            print('打开源文件失败')
            return
        if not dst_file.open(QIODevice.WriteOnly):
            print('打开目标文件失败')
            return
        dst_file.write(src_file.readAll())
        src_file.close()
        dst_file.close()
        self.update()