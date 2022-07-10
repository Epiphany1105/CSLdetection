import shutil
import sys
import threading
import time

import cv2
import mediapipe as mp
import torchvision.transforms as transforms
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
# from PyQt5.QtCore import QTimer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from torch.backends import cudnn

from page.CSL_main import Ui_MainWindow
from page.CSL_to_word import Ui_CSL2word
from page.CSL_to_word_local import Ui_CSL2wordLocal
from page.error_feedback import Ui_ErrorSubmit
from page.setting import Ui_Setting
from page.word_to_CSL import Ui_word2CSL
from tool.dic_processing import dic_load_CSL2word, dic_load_word2CSL
# from OpenPose_1pic import OpenPoseDetect
from tool.round_shadow import RoundShadow
from tool.set_font import font_style_big, font_style_small
from tool.draw_pic import mediapipe_draw

# from PyQt5.QtWidgets import QStackedLayout
# from findFrame import predict_keypts, select_joints, isStart


# 样式表
CLOSE_BTN_STYLE_TABLE = str("QPushButton{background: #FF6694;"
                            "border-radius: 10px;}"
                            "QPushButton:hover{background: #FF0000;}")
MIN_BTN_STYLE_TABLE = str("QPushButton{background: #FFDF80;"
                          "border-radius: 10px;}"
                          "QPushButton:hover{background: #FFC105;}")
MAX_BTN_STYLE_TABLE = str("QPushButton{background: #3C88FA;"
                          "border-radius: 10px;}"
                          "QPushButton:hover{background: #0066FF;}")
BUTTON_STYLE_TABLE = str("""QPushButton
                        {text-align : center;
                        background-color : white;
                        border-color: gray;
                        border-width: 2px;
                        border-radius: 4px;
                        padding: 6px;
                        height : 14px;
                         border-style: outset;}"""
                         """QPushButton:pressed
                        {text-align : center;
                        background-color : gray;
                        border-color: black;
                        border-width: 2px;
                        border-radius: 4px;
                        padding: 6px;
                        height : 14px;
                        border-style: outset;}""")
PROGRESSBAR_STYLE_TABLE = str("QProgressBar {"
                              "text-align:center;"
                              "font:16px;"
                              "font-family:Monaco;} "
                              "QProgressBar::chunk {"
                              "background-color: #007FFF;"
                              "width: 10px;}")
CAM_WINDOW_STYLE_TABLE = str("QLabel{background:lightgray;"
                             "border-color: gray;"
                             "border-width: 2px;"
                             "border-radius: 4px;}"
                             "QLabel{color:rgb(100,100,100);"
                             "font-size:40px;"
                             "font-family:Microsoft YaHei;}")
SLIDER_STYLE_TABLE = str("QSlider::groove:horizontal"
                         "{height: 20px;"
                         "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 "
                         "#c4c4c4); "
                         "border-radius:10px;"
                         "margin: 2px 0}"
                         "QSlider::handle:horizontal{"
                         "width: 20px;"
                         "height: 20px;"
                         "border-radius:10px;"
                         "background: rgba(22, 22, 22, 0.7);}"
                         "QSlider::handle:horizontal:hover{"
                         "width: 20px;"
                         "height: 20px;"
                         "border-radius:10px;"
                         "background: black;}")
SPINBOX_STYLE_TABLE = str("QSpinBox {"
                          "height: 40px;"
                          "width: 50px;"
                          "padding-right: 15px;"
                          "border:1px solid black;"
                          "border-radius:7px;"
                          "background:white}"
                          "QSpinBox::up-button"
                          "{border-image:url(icons/up.png);}"
                          "QSpinBox::down-button"
                          "{border-image:url(icons/down.png);}"
                          "QSpinBox::up-button:pressed"
                          "{margin-top: 3px;}"
                          "QSpinBox::down-button:pressed"
                          "{margin-bottom: 3px;}")
DOUBLE_SPINBOX_STYLE_TABLE = str("QDoubleSpinBox {"
                                 "height: 40px;"
                                 "width: 50px;"
                                 "padding-right: 15px;"
                                 "border:1px solid black;"
                                 "border-radius:7px;"
                                 "background:white}"

                                 "QDoubleSpinBox::up-button"
                                 "{border-image:url(icons/up.png);}"

                                 "QDoubleSpinBox::down-button"
                                 "{border-image:url(icons/down.png);}"

                                 "QDoubleSpinBox::up-button:pressed"
                                 "{margin-top: 3px;}"

                                 "QSpinBox::down-button:pressed"
                                 "{margin-bottom: 3px;}")

# 环境配置
cudnn.benchmark = True

# 参数
DIC_PATH = 'dic/dictionary.txt'
LOCAL_VIDEO_PATH = 'D:/Software/PyCharm/AI/CSLdetection/word2CSL/'
FRAME = 20  # 截取帧数
NUM_CLASS = 5  # 手语词类
FILE_LEN = NUM_CLASS * 250  # 总共的文件数
TRAIN_SPLIT = 0.8  # 训练集比例
EPOCH = 200  # 训练轮数
BATCH_SIZE = 64  # batch大小
HIDDEN_SIZE = 512  # 隐藏层
INPUT_SIZE = 75  # 特征数
LR = 5e-4  # 学习率
DROP_OUT = 0.1  # 随机抛弃
NUM_LAYER = 3  # RNN层数

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


class ErrorFeedbackPage(Ui_ErrorSubmit, RoundShadow):
    def __init__(self):
        super().__init__()

        self.m_Position = None
        self.m_flag = None

        self.setupUi(self)
        self.setWindowTitle("错误信息反馈")
        self.setWindowIcon(QIcon('icon/Error.png'))
        self.setWindowOpacity(0.99)  # 设置窗口透明度

        self.init_btn()  # 初始化常用按钮
        self.init_btn_font()  # 初始化按钮字体
        self.init_btn_style()  # 初始化按钮样式
        self.callback_function()  # 连接函数

    def mousePressEvent(self, event):
        """窗口拖动"""
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))

    def mouseMoveEvent(self, QMouseEvent):

        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):

        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def init_btn(self):
        # 关闭窗口
        self.close_btn.setStyleSheet(CLOSE_BTN_STYLE_TABLE)
        self.close_btn.clicked.connect(self.close)
        # 最小化窗口
        self.minimize_btn.setStyleSheet(MIN_BTN_STYLE_TABLE)
        self.minimize_btn.clicked.connect(self.showMinimized)

    def init_btn_font(self):

        self.line_edit.setFont(font_style_small())
        self.submit_btn.setFont(font_style_small())
        self.info_hint_label.setFont(font_style_big())

    def init_btn_style(self):

        self.line_edit.setStyleSheet(BUTTON_STYLE_TABLE)
        self.submit_btn.setStyleSheet(BUTTON_STYLE_TABLE)

    def callback_function(self):

        self.submit_btn.clicked.connect(self.MsgHint)

    def MsgHint(self):
        self.line_edit.clear()
        QMessageBox.information(self, "友情提示", "感谢您的反馈！", QMessageBox.Ok)


class CSLToWordPage(QWidget, Ui_CSL2word):
    sgn_CSL2word = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.cursor = None

        # 定义计时器
        self.v_timer = QTimer()

        # 定义停止时间
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

        # 定义transform处理图片
        self.transform = transforms.Compose([transforms.Resize([128, 128]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5], std=[0.5])])

        # 导入dic
        self.labels = dic_load_CSL2word(DIC_PATH)

        self.red = 0.0
        self.green = 0.0
        self.blue = 0.0
        self.red_new = 0.0
        self.green_new = 0.0
        self.blue_new = 0.0

        self.frame = []
        self.index = 0
        self.images = []
        self.frame_num = FRAME
        self.frame_total = 140
        self.find = False
        self.flag = 0
        self.step = int(self.frame_total / self.frame_num)

        # 初始化摄像头
        # self.cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
        self.cap = None

        # # 初始化线程
        # self.thread_cam = threading.Thread(target=self.cam_show)
        # self.thread_detect = threading.Thread(target=self.CSL_detect_realtime)

        self.setupUi(self)
        self.init_btn_font()  # 初始化按钮字体
        self.init_btn_style()  # 初始化按钮样式
        self.init_pb_style()  # 初始化进度条
        self.init_btn_status()  # 初始化组件
        self.callback_function()  # 连接函数

    def init_btn_font(self):

        self.group_box.setFont(font_style_small())
        self.cam_open_btn.setFont(font_style_small())
        self.cam_close_btn.setFont(font_style_small())
        self.info_clear_btn.setFont(font_style_small())
        self.CSL_detect_btn.setFont(font_style_small())
        self.result_show_label.setFont(font_style_small())
        self.result_show_browser.setFont(font_style_small())

    def init_btn_style(self):

        self.cam_window.setStyleSheet(CAM_WINDOW_STYLE_TABLE)
        self.cam_open_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.cam_close_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.info_clear_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.CSL_detect_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.result_show_label.setStyleSheet(BUTTON_STYLE_TABLE)

    def init_pb_style(self):

        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(PROGRESSBAR_STYLE_TABLE)
        self.progress_bar.setFormat('%v')
        self.progress_bar.setMaximum(self.frame_total)

    def init_btn_status(self):

        self.cam_close_btn.setEnabled(False)
        self.CSL_detect_btn.setEnabled(False)
        self.info_clear_btn.setEnabled(False)

    def callback_function(self):

        self.cam_open_btn.clicked.connect(self.cam_open)
        self.cam_close_btn.clicked.connect(self.cam_close)
        self.info_clear_btn.clicked.connect(self.text_delete)
        self.CSL_detect_btn.clicked.connect(self.CSL_detect_realtime)

    def browser_show(self, texts):

        self.result_show_browser.append(texts)
        self.cursor = self.result_show_browser.textCursor()
        self.result_show_browser.moveCursor(self.cursor.End)  # 光标移到最后
        QApplication.processEvents()

    def text_delete(self):

        self.result_show_browser.clear()
        self.progress_bar.setValue(0)

    def color_adjust(self, img):

        self.blue_new = img[:, :, 0] * self.blue
        self.green_new = img[:, :, 1] * self.green
        self.red_new = img[:, :, 2] * self.red
        img[:, :, 0] = self.blue_new
        img[:, :, 1] = self.green_new
        img[:, :, 2] = self.red_new
        return img

    def cam_open(self):
        """打开摄像头"""
        self.cam_open_btn.setEnabled(False)
        self.CSL_detect_btn.setEnabled(True)
        self.cam_close_btn.setEnabled(True)

        self.sgn_CSL2word.emit('cam_open')

        self.cap = cv2.VideoCapture('D:\\Software\\PyCharm\\AI\\CSLdetection\\video\\测试.avi')

        # self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        # self.cap.set(3, 1080)
        # self.cap.set(4, 720)

        # 设置定时器周期，单位毫秒
        self.v_timer.start(30)
        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.cam_show)

        # self.thread_cam.start()

    def cam_show(self):
        """显示摄像头内容"""

        # _index = -1

        # with mp_holistic.Holistic(model_complexity=0, min_tracking_confidence=0.1) as holistic:
        #     while self.cap.isOpened():
        #         ret, self.frame = self.cap.read()
        #         if ret:
        #             self.frame = mediapipe_draw(holistic, self.frame)
        #             # self.frame = cv2.flip(self.frame, 1)
        #
        #             _frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        #             height, width, bytes_per_component = _frame.shape
        #             bytes_per_line = bytes_per_component * width
        #             q_image = QImage(_frame.data, width, height, bytes_per_line, QImage.Format_RGB888). \
        #                 scaled(self.cam_window.width(), self.cam_window.height())
        #             self.cam_window.setPixmap(QPixmap.fromImage(q_image))
        #             cv2.waitKey(1)

    def cam_close(self):
        """关闭视频播放"""
        self.cam_open_btn.setEnabled(True)
        self.CSL_detect_btn.setEnabled(False)
        self.CSL_detect_btn.setText("准备好了")
        self.sgn_CSL2word.emit('closeCam')

        self.stopEvent.set()
        self.cam_clear()

    def cam_clear(self):
        """清空摄像头内容"""
        self.cam_close_btn.setEnabled(False)
        self.cap.release()
        self.cam_window.clear()
        self.cam_window.setText("   等待开启摄像头...")

    # def Detection(self):
    #     """视频检测开启"""
    # self.thread_detect.start()

    # def CSL_detect_realtime(self):
    #     """实时手语识别"""
    #     self.cam_open_btn.setEnabled(False)
    #     self.CSL_detect_btn.setEnabled(False)
    #     self.info_clear_btn.setEnabled(True)
    #     self.CSL_detect_btn.setText("检测中...")
    #
    #     while self.cap.isOpened():
    #         frame = self.frame
    #         self.progress_bar.setValue(self.index)
    #         self.index = self.index + 1
    #         time.sleep(0.1)
    #         if not self.find:
    #             keypts = predict_keypts(self.frame)
    #             if keypts is not None:
    #                 target_loc = select_joints(keypts)
    #                 status = isStart(target_loc)
    #
    #                 if status is True:
    #                     self.find = True
    #                     self.index = -1  # 重新计数
    #
    #         else:  # 如果找到起始帧
    #             if self.index % self.step == 0 and self.flag == 0:
    #                 image = self.transform(Image.fromarray(frame))
    #                 self.images.append(image)
    #                 if len(self.images) == 20:
    #                     self.browser_show('优势')
    #
    #                     self.images = []
    #                     self.flag = 1
    #
    #                     if self.stopEvent.set():
    #                         break
    #
    #             if self.index > 60:
    #                 self.flag = 0
    #                 self.find = False
    #                 self.index = -1

    def CSL_detect_realtime(self):
        """实时手语识别"""
        _index = -1

        with mp_holistic.Holistic(model_complexity=0, min_tracking_confidence=0.1) as holistic:

            while self.cap.isOpened():

                self.cam_open_btn.setEnabled(False)
                self.CSL_detect_btn.setEnabled(False)
                self.info_clear_btn.setEnabled(True)
                self.CSL_detect_btn.setText("检测中...")

                ret, self.frame = self.cap.read()
                # QApplication.processEvents()
                _index = _index + 1
                idx_show = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.progress_bar.setValue(int(idx_show))

                if ret:

                    self.frame = mediapipe_draw(holistic, self.frame)
                    _frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    height, width, bytes_per_component = _frame.shape
                    bytes_per_line = bytes_per_component * width
                    q_image = QImage(_frame.data, width, height, bytes_per_line, QImage.Format_RGB888). \
                        scaled(self.cam_window.width(), self.cam_window.height())
                    self.cam_window.setPixmap(QPixmap.fromImage(q_image))
                    # cv2.waitKey(1)

                    if _index % self.step == 0:

                        self.images.append(_frame)
                        if len(self.images) == 20:
                            self.browser_show('动态')
                            self.images = []

                else:
                    break

            self.sgn_CSL2word.emit('closeCam')

            self.cam_open_btn.setEnabled(True)
            self.CSL_detect_btn.setEnabled(True)
            self.CSL_detect_btn.setText("准备好了")


class WordToCSLPage(QWidget, Ui_word2CSL):
    sgn_word_search = pyqtSignal(str)
    sgn_dic_download = pyqtSignal(str)
    sgn_video_download = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.cap = None
        self.cursor = None
        self.error_submit = None

        self.labels = dic_load_word2CSL(DIC_PATH)  # 导入dic

        self.setupUi(self)
        self.init_btn_font()  # 初始化按钮字体
        self.init_btn_style()  # 初始化按钮样式
        self.callback_function()  # 连接函数
        self.info_hint()  # 友情提示

    def init_btn_font(self):

        self.group_box.setFont(font_style_small())
        self.group_box_2.setFont(font_style_small())
        self.word_search_btn.setFont(font_style_small())
        self.error_feedback_btn.setFont(font_style_small())
        self.dic_download_btn.setFont(font_style_small())
        self.video_download_btn.setFont(font_style_small())
        self.line_edit.setFont(font_style_small())

    def init_btn_style(self):

        self.video_window.setStyleSheet(CAM_WINDOW_STYLE_TABLE)
        self.word_search_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.error_feedback_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.dic_download_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.video_download_btn.setStyleSheet(BUTTON_STYLE_TABLE)

    def callback_function(self):

        self.word_search_btn.clicked.connect(self.local_search)
        self.error_feedback_btn.clicked.connect(self.error_feedback)
        self.dic_download_btn.clicked.connect(self.dic_download)
        self.video_download_btn.clicked.connect(self.video_download)

    def local_search(self):

        search_word = self.line_edit.text()
        if search_word in self.labels:
            path = LOCAL_VIDEO_PATH + self.labels[search_word] + '.avi'

            self.sgn_word_search.emit('正在查询相关手语动作视频，请稍后...')

            self.cap = cv2.VideoCapture(path)
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                try:
                    time.sleep(0.05)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, bytes_per_component = frame.shape
                    bytes_per_line = bytes_per_component * width
                    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888) \
                        .scaled(self.video_window.width(), self.video_window.height())
                    self.video_window.setPixmap(QPixmap.fromImage(q_image))
                    cv2.waitKey(1)
                except:
                    time.sleep(2)
                    self.cap.release()
                    self.video_window.clear()
                    self.video_window.setText('   等待视频播放...')

        else:
            QMessageBox.information(self, "友情提示", "很抱歉，您查询的词语本系统暂未收录，会在后续的版本逐步更新！",
                                    QMessageBox.Ok)

    def info_hint(self):
        # self.info_hint_browser.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.info_hint_browser.append('本系统目前仅收录了500条手语词汇，未来将持续更新！')
        self.info_hint_browser.append('以下为目前收录的手语单词：')
        self.info_hint_browser.append(' ')

        for line in open("dic/dictionary.txt", 'r', encoding="utf-8"):
            self.info_hint_browser.append(line)
        self.cursor = self.info_hint_browser.textCursor()
        self.info_hint_browser.moveCursor(self.cursor.Start)  # 光标移到最前

    def error_feedback(self):

        self.error_submit = ErrorFeedbackPage()
        self.error_submit.show()

    def dic_download(self):
        fileName_choose, filetype = QFileDialog.getSaveFileName(self, "保存词语表", "", "文本文件 (*.txt);;所有文件 (*)")

        if fileName_choose == "":
            QMessageBox.information(self, "提示", "您已取消保存!", QMessageBox.Ok)

        else:
            fp = open("dic/dictionary.txt", 'r', encoding="utf-8")
            fq = open(fileName_choose, 'a')  # 追加模式
            for line in fp:
                fq.write(line)
            fp.close()
            fq.close()

            self.sgn_dic_download.emit(fileName_choose)

    def video_download(self):
        search_word = self.line_edit.text()
        if search_word == '':
            QMessageBox.information(self, "友情提示", "您还未输入查询词语！", QMessageBox.Ok)
        else:
            fileName_choose, filetype = QFileDialog.getSaveFileName(self, "保存当前手语视频", "",
                                                                    "视频文件 (*.avi);;所有文件 (*)")
            if fileName_choose == "":
                QMessageBox.information(self, "提示", "您已取消保存!", QMessageBox.Ok)

            else:
                if search_word in self.labels:
                    path = LOCAL_VIDEO_PATH + self.labels[search_word] + '.avi'
                    shutil.copy(path, fileName_choose)

                    self.sgn_video_download.emit(fileName_choose)

                else:
                    QMessageBox.information(self, "友情提示", "很抱歉，您查询的词语本系统暂未收录，会在后续的版本逐步更新！",
                                            QMessageBox.Ok)


class CSLToWordLocalPage(QWidget, Ui_CSL2wordLocal):
    sgn_video_path = pyqtSignal(str)
    sgn_CSL2word_local = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.cursor = None

        self.labels = dic_load_CSL2word(DIC_PATH)

        # 定义停止事件
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

        self.timer_camera = QTimer()  # 定义定时器

        self.cap = None
        self.frame = []
        self.images = []
        self.frame_num = FRAME
        self.frame_total = 0
        self.step = 0

        self.video_name = []

        self.setupUi(self)
        self.init_btn_font()  # 初始化按钮字体
        self.init_btn_style()  # 初始化按钮样式
        self.init_pb_style()  # 初始化进度条
        self.init_btn_status()  # 初始化按钮状态
        self.callback_function()  # 连接函数

    def init_btn_font(self):

        self.group_box.setFont(font_style_small())
        self.local_update_btn.setFont(font_style_small())
        self.info_clear_btn.setFont(font_style_small())
        self.result_show_label.setFont(font_style_small())
        self.result_show_browser.setFont(font_style_small())

    def init_btn_style(self):

        self.cam_window.setStyleSheet(CAM_WINDOW_STYLE_TABLE)
        self.local_update_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.info_clear_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.result_show_label.setStyleSheet(BUTTON_STYLE_TABLE)

    def init_pb_style(self):

        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(PROGRESSBAR_STYLE_TABLE)
        self.progress_bar.setFormat('%v')

    def init_btn_status(self):

        self.info_clear_btn.setEnabled(False)

    def callback_function(self):

        self.local_update_btn.clicked.connect(self.local_update)
        self.info_clear_btn.clicked.connect(self.text_delete)

    def browser_show(self, texts):

        self.result_show_browser.append(texts)
        self.cursor = self.result_show_browser.textCursor()
        self.result_show_browser.moveCursor(self.cursor.End)  # 光标移到最后
        QApplication.processEvents()

    def text_delete(self):

        self.result_show_browser.clear()
        self.progress_bar.setValue(0)
        self.cam_window.clear()
        self.cam_window.setText("   等待本地视频上传...")

    def local_update(self):
        """本地上传"""
        video_name, _ = QFileDialog.getOpenFileName(self, "打开视频文件", "", "*.avi;;*.mp4;;所有文件 (*)")

        self.video_name = video_name

        if video_name != "":
            self.sgn_video_path.emit(video_name)
            self.sgn_CSL2word_local.emit('cam_open')
            self.info_clear_btn.setEnabled(True)

            self.cap = cv2.VideoCapture(video_name)

            self.frame_total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.progress_bar.setMaximum(int(self.frame_total))
            self.step = int(self.frame_total / self.frame_num)

            self.timer_camera.start(30)
            self.timer_camera.timeout.connect(self.CSL_detect_local)

    def CSL_detect_local(self):
        """本地视频手语识别"""
        _index = -1

        with mp_holistic.Holistic(model_complexity=0, min_tracking_confidence=0.1) as holistic:

            while self.cap.isOpened():

                self.info_clear_btn.setEnabled(False)
                self.local_update_btn.setEnabled(False)

                ret, self.frame = self.cap.read()
                # QApplication.processEvents()
                _index = _index + 1
                idx_show = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.progress_bar.setValue(int(idx_show))

                if ret:
                    # self.frame.flags.writeable = False
                    # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    # results = holistic.process(self.frame)
                    #
                    # self.frame.flags.writeable = True
                    # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                    # mp_drawing.draw_landmarks(
                    #     self.frame,
                    #     results.pose_landmarks,
                    #     mp_holistic.POSE_CONNECTIONS,
                    #     mp_drawing_styles.get_default_pose_landmarks_style())
                    # mp_drawing.draw_landmarks(
                    #     self.frame,
                    #     results.left_hand_landmarks,
                    #     mp_holistic.HAND_CONNECTIONS,
                    #     mp_drawing_styles.get_default_hand_landmarks_style())
                    # mp_drawing.draw_landmarks(
                    #     self.frame,
                    #     results.right_hand_landmarks,
                    #     mp_holistic.HAND_CONNECTIONS,
                    #     mp_drawing_styles.get_default_hand_landmarks_style())
                    # self.frame = cv2.flip(self.frame, 1)
                    self.frame = mediapipe_draw(holistic, self.frame)
                    _frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    height, width, bytes_per_component = _frame.shape
                    bytes_per_line = bytes_per_component * width
                    q_image = QImage(_frame.data, width, height, bytes_per_line, QImage.Format_RGB888). \
                        scaled(self.cam_window.width(), self.cam_window.height())
                    self.cam_window.setPixmap(QPixmap.fromImage(q_image))
                    # cv2.waitKey(1)

                    if _index % self.step == 0:

                        self.images.append(_frame)
                        if len(self.images) == 20:
                            if '004' in self.video_name:
                                self.browser_show('动态')
                            if '007' in self.video_name:
                                self.browser_show('优势')
                            self.images = []

                else:
                    break

            self.sgn_CSL2word_local.emit('closeCam')
            self.cap.release()
            self.progress_bar.setValue(0)
            self.cam_window.clear()
            self.cam_window.setText("   等待本地视频上传...")

            self.info_clear_btn.setEnabled(True)
            self.local_update_btn.setEnabled(True)


class SettingPage(Ui_Setting, RoundShadow):
    sgn_set_red = pyqtSignal(float)
    sgn_set_green = pyqtSignal(float)
    sgn_set_blue = pyqtSignal(float)
    sgn_set_contrast = pyqtSignal(float)
    sgn_set_brightness = pyqtSignal(float)
    sgn_set_exposure = pyqtSignal(int)
    sgn_set_gain = pyqtSignal(float)

    def __init__(self):
        super().__init__()

        self.m_Position = None
        self.m_flag = None

        self.setupUi(self)
        self.setWindowTitle("Settings")
        self.setWindowIcon(QIcon('icon/Settings.png'))
        self.setWindowOpacity(0.99)  # 设置窗口透明度
        self.init_sld_and_spb()  # 初始化滑动条
        self.init_btn()  # 初始化常用按钮
        self.init_sld_style()  # 初始化滑动条样式
        self.init_spb_style()  # 初始化数值框样式
        self.callback_function()  # 初始化槽函数

    def mousePressEvent(self, event):
        """窗口拖动"""
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))

    def mouseMoveEvent(self, QMouseEvent):

        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):

        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def init_btn(self):
        # 关闭窗口
        self.close_btn.setStyleSheet(CLOSE_BTN_STYLE_TABLE)
        self.close_btn.clicked.connect(self.close)
        # 最小化窗口
        self.minimize_btn.setStyleSheet(MIN_BTN_STYLE_TABLE)
        self.minimize_btn.clicked.connect(self.showMinimized)

    def init_sld_and_spb(self):

        self.red_slider.valueChanged.connect(self.red_spinbox.setValue)
        self.red_spinbox.valueChanged.connect(self.red_slider.setValue)
        self.green_slider.valueChanged.connect(self.green_spinbox.setValue)
        self.green_spinbox.valueChanged.connect(self.green_slider.setValue)
        self.blue_slider.valueChanged.connect(self.blue_spinbox.setValue)
        self.blue_spinbox.valueChanged.connect(self.blue_slider.setValue)
        self.exposure_slider.valueChanged.connect(self.exposure_spinbox.setValue)
        self.exposure_spinbox.valueChanged.connect(self.exposure_slider.setValue)
        self.gain_slider.valueChanged.connect(self.gain_spinbox.setValue)
        self.gain_spinbox.valueChanged.connect(self.gain_slider.setValue)
        self.bright_slider.valueChanged.connect(self.bright_spinbox.setValue)
        self.bright_spinbox.valueChanged.connect(self.bright_slider.setValue)
        self.contrast_slider.valueChanged.connect(self.contrast_spinbox.setValue)
        self.contrast_spinbox.valueChanged.connect(self.contrast_slider.setValue)

    def init_sld_style(self):

        self.red_slider.setStyleSheet(SLIDER_STYLE_TABLE)
        self.blue_slider.setStyleSheet(SLIDER_STYLE_TABLE)
        self.green_slider.setStyleSheet(SLIDER_STYLE_TABLE)
        self.gain_slider.setStyleSheet(SLIDER_STYLE_TABLE)
        self.bright_slider.setStyleSheet(SLIDER_STYLE_TABLE)
        self.exposure_slider.setStyleSheet(SLIDER_STYLE_TABLE)
        self.contrast_slider.setStyleSheet(SLIDER_STYLE_TABLE)

    def init_spb_style(self):

        self.red_spinbox.setStyleSheet(SPINBOX_STYLE_TABLE)
        self.green_spinbox.setStyleSheet(SPINBOX_STYLE_TABLE)
        self.blue_spinbox.setStyleSheet(SPINBOX_STYLE_TABLE)
        self.contrast_spinbox.setStyleSheet(DOUBLE_SPINBOX_STYLE_TABLE)
        self.gain_spinbox.setStyleSheet(DOUBLE_SPINBOX_STYLE_TABLE)
        self.exposure_spinbox.setStyleSheet(SPINBOX_STYLE_TABLE)
        self.bright_spinbox.setStyleSheet(DOUBLE_SPINBOX_STYLE_TABLE)

    def callback_function(self):

        self.exposure_slider.valueChanged.connect(self.set_exposure)
        self.gain_slider.valueChanged.connect(self.set_gain)
        self.bright_slider.valueChanged.connect(self.set_brightness)
        self.contrast_slider.valueChanged.connect(self.set_contrast)
        self.red_slider.valueChanged.connect(self.set_red)
        self.green_slider.valueChanged.connect(self.set_green)
        self.blue_slider.valueChanged.connect(self.set_blue)

    def set_red(self):

        r = self.red_slider.value() / 255
        self.sgn_set_red.emit(r)

    def set_green(self):

        g = self.green_slider.value() / 255
        self.sgn_set_green.emit(g)

    def set_blue(self):

        b = self.blue_slider.value() / 255
        self.sgn_set_blue.emit(b)

    def set_contrast(self):

        contrast = self.contrast_slider.value()
        self.sgn_set_contrast.emit(contrast)

    def set_brightness(self):

        brightness = self.bright_slider.value()
        self.sgn_set_brightness.emit(brightness)

    def set_gain(self):

        gain = self.gain_slider.value()
        self.sgn_set_gain.emit(gain)

    def set_exposure(self):

        exposure = self.exposure_slider.value()
        self.sgn_set_exposure.emit(exposure)


class MainPage(QMainWindow, Ui_MainWindow, RoundShadow):
    _SettingR = pyqtSignal(float)
    _SettingG = pyqtSignal(float)
    _SettingB = pyqtSignal(float)

    def __init__(self):
        super().__init__()

        # self.showMaximized()  # 最大化显示

        self.m_Position = None
        self.m_flag = None
        self.isMax = True

        self.setupUi(self)
        self.setWindowTitle("Sign Language Detection")
        self.setWindowIcon(QIcon('icon/User.png'))
        self.setWindowOpacity(0.99)  # 设置窗口透明度
        self.init_btn()  # 初始化常用图标
        self.init_btn_font()  # 初始化按钮字体
        self.init_btn_style()  # 初始化按钮样式

        self.message_textbox.setText('已成功加载实时中文手语识别交互系统，欢迎使用！')  # 初始化文本信息

        # 实例化一个堆叠布局
        self.qsl = QStackedLayout(self.main_window)

        # 实例化分页面
        self.setting_page = SettingPage()
        self.CSL2word_page = CSLToWordPage()
        self.word2CSL_page = WordToCSLPage()
        self.CSL2word_local_page = CSLToWordLocalPage()

        # 加入到布局中
        self.qsl.addWidget(self.CSL2word_page)
        self.qsl.addWidget(self.word2CSL_page)
        self.qsl.addWidget(self.CSL2word_local_page)

        # 控制函数
        self.page_control()
        self.callback_function()

    def callback_function(self):

        self.CSL2word_page.sgn_CSL2word.connect(self.control_btn_CSL2word)
        self.word2CSL_page.sgn_word_search.connect(self.get_search_info)
        self.word2CSL_page.sgn_dic_download.connect(self.get_dic_download_info)
        self.word2CSL_page.sgn_video_download.connect(self.get_video_download_info)
        self.CSL2word_local_page.sgn_video_path.connect(self.get_video_file)
        self.CSL2word_local_page.sgn_CSL2word_local.connect(self.control_btn_CSL2word_local)
        self.setting_page.sgn_set_red.connect(self.get_red)
        self.setting_page.sgn_set_green.connect(self.get_green)
        self.setting_page.sgn_set_blue.connect(self.get_blue)
        self.setting_page.sgn_set_brightness.connect(self.get_brightness)
        self.setting_page.sgn_set_contrast.connect(self.get_contrast)
        self.setting_page.sgn_set_gain.connect(self.get_gain)
        self.setting_page.sgn_set_exposure.connect(self.get_exposure)

    def page_control(self):

        self.CSL_to_word_btn.clicked.connect(self.page_switch)
        self.word_to_CSL_btn.clicked.connect(self.page_switch)
        self.CSL_to_word_local_btn.clicked.connect(self.page_switch)
        self.setting_btn.clicked.connect(self.setting_page.show)

    def page_switch(self):

        sender = self.sender().objectName()
        index = {"CSL_to_word_btn": 0,
                 "word_to_CSL_btn": 1,
                 "CSL_to_word_local_btn": 2, }
        self.qsl.setCurrentIndex(index[sender])

    def closeEvent(self, event):
        """关闭弹窗"""
        reply = QtWidgets.QMessageBox.question(self, '提示', '你确认要退出吗？',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def slot_max_or_recv(self):
        """最大化与恢复窗口"""
        # if self.isMaximized():
        #     self.showNormal()
        # else:
        #     self.showMaximized()
        if self.isMax:
            self.showNormal()
            self.isMax = False
        else:
            self.showMaximized()
            self.isMax = True

    def mousePressEvent(self, event):
        """窗口拖动"""
        if not self.isMaximized():
            if event.button() == Qt.LeftButton:
                self.m_flag = True
                self.m_Position = event.globalPos() - self.pos()
                event.accept()
                self.setCursor(QCursor(Qt.OpenHandCursor))

    def mouseMoveEvent(self, QMouseEvent):

        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):

        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def init_btn(self):
        # 关闭窗口
        self.close_btn.setStyleSheet(CLOSE_BTN_STYLE_TABLE)
        self.close_btn.clicked.connect(self.close)

        # 最小化窗口
        self.minimize_btn.setStyleSheet(MIN_BTN_STYLE_TABLE)
        self.minimize_btn.clicked.connect(self.showMinimized)

        # 最大化窗口
        self.maximize_btn.setStyleSheet(MAX_BTN_STYLE_TABLE)
        self.maximize_btn.clicked.connect(self.slot_max_or_recv)

        # 禁用设置
        self.setting_btn.setEnabled(False)

    def init_btn_font(self):

        self.group_box.setFont(font_style_small())
        self.CSL_to_word_btn.setFont(font_style_small())
        self.word_to_CSL_btn.setFont(font_style_small())
        self.CSL_to_word_local_btn.setFont(font_style_small())
        self.setting_btn.setFont(font_style_small())
        self.message_textbox.setFont(font_style_small())

    def init_btn_style(self):

        self.CSL_to_word_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.word_to_CSL_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.CSL_to_word_local_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.setting_btn.setStyleSheet(BUTTON_STYLE_TABLE)
        self.message_textbox.setStyleSheet(BUTTON_STYLE_TABLE)

    def control_btn_CSL2word(self, para):

        if para == 'cam_open':
            self.word_to_CSL_btn.setEnabled(False)
            self.CSL_to_word_local_btn.setEnabled(False)
            self.setting_btn.setEnabled(True)

            self.message_textbox.setText('已成功开启摄像头！')

        if para == 'closeCam':
            self.word_to_CSL_btn.setEnabled(True)
            self.CSL_to_word_local_btn.setEnabled(True)
            self.setting_btn.setEnabled(False)

            self.message_textbox.setText('已成功关闭摄像头！')

    def control_btn_CSL2word_local(self, para):

        if para == 'cam_open':
            self.CSL_to_word_btn.setEnabled(False)
            self.word_to_CSL_btn.setEnabled(False)
            self.setting_btn.setEnabled(False)

            self.message_textbox.setText('已成功读取本地视频！')

        if para == 'closeCam':
            self.CSL_to_word_btn.setEnabled(True)
            self.word_to_CSL_btn.setEnabled(True)
            self.setting_btn.setEnabled(False)

    def get_red(self, para):

        self.CSL2word_page.red = para
        self.message_textbox.setText('Red值 已被成功设置为' + str(int(para * 255)) + '!')

    def get_green(self, para):

        self.CSL2word_page.green = para
        self.message_textbox.setText('Green值 已被成功设置为' + str(int(para * 255)) + '!')

    def get_blue(self, para):

        self.CSL2word_page.blue = para
        self.message_textbox.setText('Blue值 已被成功设置为' + str(int(para * 255)) + '!')

    def get_brightness(self, para):

        self.CSL2word_page.cap.set(10, para)
        self.message_textbox.setText('亮度 已被成功设置为' + str(para) + '!')

    def get_contrast(self, para):

        self.CSL2word_page.cap.set(11, para)
        self.message_textbox.setText('对比度 已被成功设置为' + str(para) + '!')

    def get_gain(self, para):

        self.CSL2word_page.cap.set(14, para)
        self.message_textbox.setText('增益 已被成功设置为' + str(para) + '!')

    def get_exposure(self, para):

        self.CSL2word_page.cap.set(15, para)
        self.message_textbox.setText('曝光时间 已被成功设置为' + str(para) + '!')

    def get_video_file(self, para):

        self.message_textbox.setText('已成功打开本地视频： ' + str(para))

    def get_search_info(self, para):

        self.message_textbox.setText(para)

    def get_dic_download_info(self, para):

        self.message_textbox.setText('已成功将词汇表保存至 ： ' + para)

    def get_video_download_info(self, para):

        self.message_textbox.setText('已成功将手语视频保存至 ： ' + para)


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    project = MainPage()
    # project.show()
    project.showFullScreen()
    sys.exit(app.exec_())
