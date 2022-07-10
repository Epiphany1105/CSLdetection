from PyQt5 import QtGui


def font_style_big():
    font = QtGui.QFont()
    font.setFamily("Microsoft YaHei")
    font.setPixelSize(18)
    font.setBold(True)
    font.setWeight(75)
    return font


def font_style_small():
    font = QtGui.QFont()
    font.setFamily("Microsoft YaHei")
    font.setPixelSize(15)
    font.setBold(True)
    font.setWeight(75)
    return font


def font_style_in_setting():
    font = QtGui.QFont()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(13)
    font.setBold(True)
    font.setWeight(75)
    return font
