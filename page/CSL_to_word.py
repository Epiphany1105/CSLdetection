# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CSL_to_word.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CSL2word(object):
    def setupUi(self, CSL2word):
        CSL2word.setObjectName("CSL2word")
        CSL2word.resize(889, 613)
        self.gridLayout_2 = QtWidgets.QGridLayout(CSL2word)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.cam_window = QtWidgets.QLabel(CSL2word)
        self.cam_window.setMinimumSize(QtCore.QSize(0, 0))
        self.cam_window.setObjectName("cam_window")
        self.gridLayout.addWidget(self.cam_window, 0, 1, 3, 1)
        self.group_box = QtWidgets.QGroupBox(CSL2word)
        self.group_box.setMinimumSize(QtCore.QSize(0, 220))
        self.group_box.setMaximumSize(QtCore.QSize(165, 220))
        self.group_box.setObjectName("group_box")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.group_box)
        self.verticalLayout.setObjectName("verticalLayout")
        self.cam_open_btn = QtWidgets.QPushButton(self.group_box)
        self.cam_open_btn.setMinimumSize(QtCore.QSize(100, 40))
        self.cam_open_btn.setMaximumSize(QtCore.QSize(200, 60))
        self.cam_open_btn.setObjectName("cam_open_btn")
        self.verticalLayout.addWidget(self.cam_open_btn)
        self.CSL_detect_btn = QtWidgets.QPushButton(self.group_box)
        self.CSL_detect_btn.setMinimumSize(QtCore.QSize(100, 40))
        self.CSL_detect_btn.setMaximumSize(QtCore.QSize(200, 60))
        self.CSL_detect_btn.setObjectName("CSL_detect_btn")
        self.verticalLayout.addWidget(self.CSL_detect_btn)
        self.cam_close_btn = QtWidgets.QPushButton(self.group_box)
        self.cam_close_btn.setMinimumSize(QtCore.QSize(100, 40))
        self.cam_close_btn.setMaximumSize(QtCore.QSize(200, 60))
        self.cam_close_btn.setObjectName("cam_close_btn")
        self.verticalLayout.addWidget(self.cam_close_btn)
        self.info_clear_btn = QtWidgets.QPushButton(self.group_box)
        self.info_clear_btn.setMinimumSize(QtCore.QSize(100, 40))
        self.info_clear_btn.setMaximumSize(QtCore.QSize(200, 60))
        self.info_clear_btn.setObjectName("info_clear_btn")
        self.verticalLayout.addWidget(self.info_clear_btn)
        self.gridLayout.addWidget(self.group_box, 0, 0, 1, 1)
        self.result_show_label = QtWidgets.QLabel(CSL2word)
        self.result_show_label.setMinimumSize(QtCore.QSize(0, 0))
        self.result_show_label.setMaximumSize(QtCore.QSize(165, 40))
        self.result_show_label.setObjectName("result_show_label")
        self.gridLayout.addWidget(self.result_show_label, 1, 0, 1, 1)
        self.progress_bar = QtWidgets.QProgressBar(CSL2word)
        self.progress_bar.setMinimumSize(QtCore.QSize(220, 15))
        self.progress_bar.setMaximumSize(QtCore.QSize(16777215, 30))
        self.progress_bar.setProperty("value", 24)
        self.progress_bar.setObjectName("progress_bar")
        self.gridLayout.addWidget(self.progress_bar, 3, 1, 1, 1)
        self.result_show_browser = QtWidgets.QTextBrowser(CSL2word)
        self.result_show_browser.setMinimumSize(QtCore.QSize(0, 0))
        self.result_show_browser.setMaximumSize(QtCore.QSize(165, 16777215))
        self.result_show_browser.setObjectName("result_show_browser")
        self.gridLayout.addWidget(self.result_show_browser, 2, 0, 2, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(CSL2word)
        QtCore.QMetaObject.connectSlotsByName(CSL2word)

    def retranslateUi(self, CSL2word):
        _translate = QtCore.QCoreApplication.translate
        CSL2word.setWindowTitle(_translate("CSL2word", "Form"))
        self.cam_window.setText(_translate("CSL2word", "等待开启摄像头..."))
        self.group_box.setTitle(_translate("CSL2word", "手语识别"))
        self.cam_open_btn.setText(_translate("CSL2word", "开启摄像头"))
        self.CSL_detect_btn.setText(_translate("CSL2word", "准备好了"))
        self.cam_close_btn.setText(_translate("CSL2word", "关闭摄像头"))
        self.info_clear_btn.setText(_translate("CSL2word", "清除预测结果"))
        self.result_show_label.setText(_translate("CSL2word", "预测结果"))