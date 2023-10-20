from typing import Union

import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qframelesswindow import *

from qfluentwidgets import *
from utils.UI_clip import Widget_clip
from utils.UI_filter import Widget_filter
from utils.UI_Insert import Widget_Insert
from utils.UI_segmentation import Widget_segmentation
from utils.UI_repair import Widget_repair
from utils.UI_gif import Widget_gif
from utils.UI_effects import Widget_effects
from utils.UI_transfer import Widget_transfer
from utils.UI_modify import Widget_modify


class Window(FramelessWindow):
    # 创建打开文件的信号
    filename_signal = pyqtSignal(str)
    # 创建保存的信号
    save_signal = pyqtSignal(str)
    # 创建Reset信号
    reset_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filename = ''
        # 使用numpy(cv2)数组存储图像
        self.image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # 白色背景
        # 设置标题
        stdtitlebar = StandardTitleBar(self)
        stdtitlebar.setTitle('图像处理')
        self.setTitleBar(stdtitlebar)
        ## 设置主题色
        # setTheme(Theme.DARK)
        setThemeColor('#87CEFA')
        setTheme(Theme.LIGHT)

        # 创建最外层布局-最外层为水平布局
        self.hBoxLayout = QHBoxLayout(self)
        # 创建导航组件
        self.navigationInterface = NavigationInterface(self, showMenuButton=True)
        # 创建抽屉组件
        self.stackwidget = QStackedWidget(self)

        # 此处添加界面实例
        self.Interface_filter = Widget_filter(self)
        self.Interface_clip = Widget_clip(self)
        self.Interface_insert = Widget_Insert(self)
        self.Interface_segmentation = Widget_segmentation(self)
        self.Interface_repair = Widget_repair(self)
        self.Interface_gif = Widget_gif(self)
        self.Interface_effect = Widget_effects(self)
        self.Interface_transfer = Widget_transfer(self)
        self.Interface_modify = Widget_modify(self)


        # Todo

        # 初始化布局, 导航和窗口
        self.init_Layout()
        self.init_navigation()
        self.init_Window()

    def init_Layout(self):
        # 设置控件与窗口间距
        self.hBoxLayout.setSpacing(0)
        # 设置布局的边距, 不占用顶部标题栏的文职
        self.hBoxLayout.setContentsMargins(0, self.titleBar.height(), 0, 0)
        # 添加导航栏和stackwidget
        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addWidget(self.stackwidget)
        # 设置缩放比例
        self.hBoxLayout.setStretchFactor(self.stackwidget, 1)

    def init_navigation(self):
        """初始化导航栏"""
        path = 'resources/icon/'
        # 此处添加实例
        self.CreateInterface(self.Interface_filter, icon=FluentIcon.EDIT, text='图像滤波')
        self.CreateInterface(self.Interface_clip, icon=FluentIcon.CLIPPING_TOOL, text='图像裁剪')
        self.CreateInterface(self.Interface_insert, icon=self.GetIcon(path +'插入.png'), text='插入')
        self.CreateInterface(self.Interface_segmentation, icon=self.GetIcon(path + '图形分割.png'), text='图形分割')
        self.CreateInterface(self.Interface_repair, icon=self.GetIcon(path + '图像修复.png'), text='图像修复')
        self.CreateInterface(self.Interface_gif, icon=self.GetIcon(path + 'GIF.png'), text='GIF')
        self.CreateInterface(self.Interface_effect, icon=self.GetIcon(path + '特效.png'), text='图像特效')
        self.CreateInterface(self.Interface_transfer, icon=self.GetIcon(path + '转换.png'), text='图像转换')
        self.CreateInterface(self.Interface_modify, icon=self.GetIcon(path + '亮度.png'), text='亮度修改')
        # Todo

        # 添加打开文件选项
        self.navigationInterface.addItem(
            routeKey='openfile',
            icon=FluentIcon.FOLDER,
            text='打开文件',
            onClick=lambda: self.openfile(),
            position=NavigationItemPosition.BOTTOM,
            parentRouteKey=None
        )
        # 添加保存文件选项
        self.navigationInterface.addItem(
            routeKey='savefile',
            icon=FluentIcon.SAVE,
            text='保存文件',
            onClick=lambda: self.savefile(),
            position=NavigationItemPosition.BOTTOM,
            parentRouteKey=None
        )
        # 添加另存为选项
        self.navigationInterface.addItem(
            routeKey='saveasfile',
            icon=FluentIcon.SAVE_AS,
            text='另存为',
            onClick=lambda: self.saveasfile(),
            position=NavigationItemPosition.BOTTOM,
            parentRouteKey=None
        )

        # 添加头像
        self.navigationInterface.addItem(
            routeKey='team_info',
            icon=FluentIcon.INFO,
            text='小组信息',
            onClick=lambda: self.ShowTeamInfo(),
            position=NavigationItemPosition.BOTTOM,
            parentRouteKey=None
        )
        # 绑定信号槽
        self.stackwidget.currentChanged.connect(self.onCurrentInterfaceChanged)
        self.stackwidget.setCurrentIndex(1)

    def init_Window(self):
        self.resize(900, 700)
        # 设置窗口图标
        self.setWindowIcon(QIcon('resources/figure/IJN.png'))
        # 设置窗口标题
        self.setWindowTitle('小组1')
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        # 设置样式
        self.setQss()

    def CreateInterface(self, interface, icon: Union[str, QIcon], text: str, position=NavigationItemPosition.TOP,
                        parent=None):
        """设置导航栏item和对应的stackwidget显示内容"""
        '''
        interface: stackwidget显示内容
        txt: item的文本描述
        icon: item图标
        position: item的位置
        '''
        self.stackwidget.addWidget(interface)
        self.navigationInterface.addItem(
            routeKey=interface.objectName(),
            icon=icon,
            text=text,
            onClick=lambda: self.switchInterface(interface),
            position=position,
            tooltip=text,
            parentRouteKey=parent.objectName() if parent else None
        )

    def switchInterface(self, widget):
        """设置当前stackwidget显示的interface"""
        self.stackwidget.setCurrentWidget(widget)
        # 加载widget中的图像
        widget.Set_image(self.image)

    def onCurrentInterfaceChanged(self, index):
        """切换导航栏选定的item"""
        widget = self.stackwidget.widget(index)
        self.navigationInterface.setCurrentItem(widget.objectName())

    def setQss(self):
        color = 'dark' if isDarkTheme() else 'light'
        with open(f'resources/{color}/test.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def openfile(self):
        """打开文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "打开文件", "", "图像文件 (*.jpg *.png);;所有文件 (*)",
                                                   options=options)
        if file_name == '':
            return
        self.filename = file_name
        self.image = cv2.imread(file_name)
        # 更新当前widget显示的图片
        widget = self.stackwidget.currentWidget()
        widget.Set_image(self.image)

    def savefile(self):
        """保存文件"""
        # 将当前界面的图片保存到image对象中
        widget = self.stackwidget.currentWidget()
        self.image = widget.Get_image()

        if self.filename == '':
            return
        else:
            cv2.imwrite(self.filename, self.image)
        # 将图像保存到指定路径

    def saveasfile(self):
        """另存为"""
        # 将当前界面的图片保存到image对象中
        widget = self.stackwidget.currentWidget()
        self.image = widget.Get_image()

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "另存为", "", "图像文件 (*.jpg *.png);;所有文件 (*)",
                                                   options=options)
        if file_name == '':
            return
        else:
            cv2.imwrite(file_name, self.image)

    def ShowTeamInfo(self):
        dialog = Dialog('小组信息', 'UI设计: 58121217赵漭, 58121224严正阳\n算法设计: 58121124张博彦, 58121225沈俊燃',
                        self)
        dialog.cancelButton.hide()
        dialog.yesButton.setText('好的')
        dialog.show()

    def GetIcon(self, path=''):
        """创建图标"""
        if path != '':
            return QIcon(QPixmap(path))
        else:
            return None

    def transfer_BGR2BGRA(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec()
