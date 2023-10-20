from typing import Union

import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qframelesswindow import *

from qfluentwidgets import *

class display_GIF(QWidget):
    def __init__(self,path: str ,parent=None):
        super().__init__(parent)
        self.init_ui(path)
        self.setObjectName('display_GIF')

    def init_ui(self,path:str):
        # 创建相关组件
        self.container = QVBoxLayout(self)
        self.load_path = path
        self.gif_label = QLabel()
        self.gif_movie = QMovie(path)
        self.gif_label.setMovie(self.gif_movie)
        self.btn_close = PushButton('取消')
        self.btn_yes = PushButton('保存')
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.btn_yes)
        self.btn_layout.addWidget(self.btn_close)
        self.container.addWidget(self.gif_label)
        self.container.addLayout(self.btn_layout)
        self.gif_movie.start()

    def BindSlot(self):
        self.btn_yes.clicked.connect(self.SaveGIF)
        self.btn_close.clicked.connect(self.close)

    # def paintEvent(self, a0):
    #     super().paintEvent(a0)
    #     self.gif_movie.updated()
    #     self.gif_label.update()

    def SaveGIF(self):
        # 打开文件对话框
        options = QFileDialog.Options()
        target_file, _ = QFileDialog.getSaveFileName(self, "另存为", "", "图像文件 (*.gif);;所有文件 (*)",
                                                   options=options)
        source_file = QFile(self.load_path)
        if source_file.open(QIODevice.ReadOnly) and target_file.open(QIODevice.WriteOnly):
            # 读取源文件内容
            data = source_file.readAll()
            # 将读取的内容写入目标文件
            target_file.write(data)
            # 关闭文件
            source_file.close()
            target_file.close()
            print("文件拷贝成功！")
        else:
            print("无法打开文件")
        self.close()

