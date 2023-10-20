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


class Widget_transfer(QFrame):
    finish_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setObjectName('UI_transfer')
        self.pos_src = []
        self.pos_dst = []
        self.click_src = False
        self.click_dst = False 
        self.aim = 0

    def init_ui(self):
        #创建算法实例
        self.filter = Filter()
        self.container = QVBoxLayout(self)
        
        # 创建备份图片, 仅在读取图片和保存修改时修改内容, 在重置修改时调用
        self.image_backup = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # 白色背景
        
        # 加载显示图片类型
        self.widgetImage = Widget_Image(self.image_backup)
        
        #创建布局
        self.layout_v_label = QVBoxLayout()
        self.layout_v_edit = QVBoxLayout()
        self.layout_h_layout = QHBoxLayout()
        # 创建按钮
        self.bt_affine = PushButton('仿射变换')
        self.bt_perspective = PushButton('透视变换')
        self.bt_saveBGRA = PushButton('转换为透明底png图像')
        self.bt_savechange = PushButton('保存修改')
        self.bt_reset = PushButton('重置修改')
        
        self.display_pos_src = QLabel('已选择点(src):')
        self.display_pos_dst = QLabel('已选择点(dst):')
        self.display_layout = QVBoxLayout()
        self.display_layout.addWidget(self.display_pos_src)
        self.display_layout.addWidget(self.display_pos_dst)

        self.transfer_layout = QVBoxLayout()
        self.transfer_layout.addWidget(self.bt_affine)
        self.transfer_layout.addWidget(self.bt_perspective)
        self.transfer_layout.addWidget(self.bt_saveBGRA)

        self.layout_saveandchange = QVBoxLayout()
        self.layout_saveandchange.addWidget(self.bt_savechange)
        self.layout_saveandchange.addWidget(self.bt_reset)
     
        self.layout_outer = QHBoxLayout()
        self.layout_outer.addLayout(self.display_layout)
        self.layout_outer.addLayout(self.transfer_layout)
        self.layout_outer.addLayout(self.layout_saveandchange)
        # 设置布局
        self.container.addWidget(self.widgetImage)
        self.container.addLayout(self.layout_outer)

        self.setLayout(self.container)
        # 绑定信号槽
        self.BindSlot()

    def BindSlot(self):
        #Todo
        self.bt_affine.clicked.connect(self.Affine)
        self.bt_perspective.clicked.connect(self.Perspective)
        self.bt_reset.clicked.connect(self.Reset_image)
        self.bt_savechange.clicked.connect(self.Save_change_image)
        self.finish_signal.connect(self.handle_signal)
        self.bt_saveBGRA.clicked.connect(self.SaveBGRA)


    def Affine(self):
        self.pos_src.clear()
        self.pos_dst.clear()
        self.display_pos_dst.setText('已选择点(dst):')
        self.display_pos_src.setText('已选择点(src):')
        self.aim = 3
        self.click_src = True
        self.click_dst = False

    def Perspective(self):
        self.pos_src.clear()
        self.pos_dst.clear()
        self.display_pos_dst.setText('已选择点(dst):')
        self.display_pos_src.setText('已选择点(src):')
        self.aim = 4
        self.click_src = True
        self.click_dst = False

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
        self.display_pos_dst.setText('已选择点(dst):')
        self.display_pos_src.setText('已选择点(src):')
        self.update()

    def Save_change_image(self):
        """保存修改"""
        self.image_backup = self.Get_image()

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.widgetImage.mousePressEvent(event)
            if self.click_src:
                self.pos_src.append(list(self.widgetImage.Get_point()))
                print(self.pos_src)
                self.display_pos_src.setText('已选择点(src):' + str(self.pos_src))
                self.update()
                if len(self.pos_src) == self.aim:
                    self.click_src = False
                    self.click_dst = True
            elif self.click_dst:
                self.pos_dst.append(list(self.widgetImage.Get_point()))
                self.display_pos_dst.setText('已选择点(dst):' + str(self.pos_dst))
                self.update()
                if len(self.pos_dst) == self.aim:
                    self.click_dst = False
                    self.finish_signal.emit(self.aim)



    def handle_signal(self, aim):
        # 代表仿射变换
        src_arr = np.array(self.pos_src, dtype=np.float32)
        dst_arr = np.array(self.pos_dst, dtype=np.float32)
        if aim == 3:
            self.widgetImage.Set_image(self.filter.affine(self.Get_image(), src_arr, dst_arr))
        # 代表透视变换
        elif aim == 4:
            self.widgetImage.Set_image(self.filter.perspective(self.Get_image(), src_arr, dst_arr))
        self.update()

    def SaveBGRA(self):
        """转换为透明底png图像"""
        self.widgetImage.Set_image(self.filter.BGR2BGRA(self.Get_image()))
        self.update()