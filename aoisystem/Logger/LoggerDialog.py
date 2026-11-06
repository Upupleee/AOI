# This Python file uses the following encoding: utf-8
import logging
import sys
import re
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from PyQt5.QtGui import QTextCursor, QStandardItemModel, QStandardItem
from PyQt5.QtCore import QDateTime


class QTextEditHandler(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        self.text_edit.append(msg)


class LoggerDialog(QDialog):
    def __init__(self):
        super(LoggerDialog, self).__init__()
        self.ui = uic.loadUiType("./Logger/Logger.ui")[0]()
        self.ui.setupUi(self)
        self.count = 0
        self.ui.dateTimeEdit_start.setDateTime(QDateTime.currentDateTime())
        self.ui.dateTimeEdit_end.setDateTime(QDateTime.currentDateTime())
        self.ui.query_button.clicked.connect(self.queryLogs)
        self.ui.next_result_button.clicked.connect(self.search_next)
        self.ui.previous_result_button.clicked.connect(self.search_prev)

        # 设置日志记录器
        self.logger = logging.getLogger('aoi_system.log')
        self.logger.setLevel(logging.INFO)
        self.file_handler = logging.FileHandler('./Logger/aoi_system.log')
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.file_handler)

        # 读取日志文件内容并显示在文本编辑框中
        try:
            with open('./Logger/aoi_system.log', 'r') as log_file:
                self.log_content = log_file.read()
                self.ui.log_textedit.setPlainText(self.log_content)
        except FileNotFoundError:
            pass

        # 将日志记录器的输出重定向到文本编辑框
        self.text_edit_handler = QTextEditHandler(self.ui.log_textedit)
        self.stream_handler = logging.StreamHandler(self.ui.log_textedit.document())
        self.text_edit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.text_edit_handler)

        # 初始化时间范围变量
        self.start_date = None
        self.end_date = None

    def queryLogs(self):
        if self.ui.checkBox_datetime.isChecked():
            start_date = self.ui.dateTimeEdit_start.dateTime()
            end_date = self.ui.dateTimeEdit_end.dateTime()
            filtered_logs = [log for log in self.log_content.splitlines() if start_date <= QDateTime.fromString(log[:19], "yyyy-MM-dd hh:mm:ss") <= end_date]
            result_text = ""
            for log in filtered_logs:
                # 这里假设 log[0] 是日期时间字符串，可以直接格式化输出
                result_text += log[:19] + " - " + log[19:] + "\n"
            self.ui.log_textedit.setPlainText(result_text)
        else:
            try:
                with open('./Logger/aoi_system.log', 'r') as log_file:
                    self.log_content = log_file.read()
                    self.ui.log_textedit.setPlainText(self.log_content)
            except FileNotFoundError:
                pass
    def search_next(self):
        search_text = self.ui.search_lineEdit.text()
        main_text = self.ui.log_textedit.toPlainText()
        res_list = []
        flags = 0
        if self.ui.case_check.isChecked():
            if not self.ui.regex_check.isChecked():
                flags = flags | re.IGNORECASE
        if self.ui.regex_check.isChecked():
            try:
                pattern = re.compile(search_text, flags)
                if self.start_date and self.end_date:
                    for line in main_text.splitlines():
                        timestamp_str = line[:19]
                        timestamp = QDateTime.fromString(timestamp_str, "yyyy-MM-dd hh:mm:ss")
                        if self.start_date <= timestamp <= self.end_date:
                            for match in re.finditer(pattern, line[20:]):
                                res_list.append((match.start(), match.end()))
                else:
                    for match in re.finditer(pattern, main_text):
                        res_list.append(match.span())
            except re.error:
                QMessageBox.warning(self, '错误提示', '正则表达式错误。')
                return
        else:
            if self.start_date and self.end_date:
                for line in main_text.splitlines():
                    timestamp_str = line[:19]
                    timestamp = QDateTime.fromString(timestamp_str, "yyyy-MM-dd hh:mm:ss")
                    if self.start_date <= timestamp <= self.end_date:
                        for match in re.finditer(re.escape(search_text), line[20:], flags):
                            res_list.append((match.start(), match.end()))
            else:
                for match in re.finditer(re.escape(search_text), main_text, flags):
                    res_list.append(match.span())

        if res_list:
            try:
                res = res_list[self.count]
                tc = self.ui.log_textedit.textCursor()
                if self.start_date and self.end_date:
                    offset = 0
                    for line in main_text.splitlines():
                        if self.start_date <= QDateTime.fromString(line[:19], "yyyy-MM-dd hh:mm:ss") <= self.end_date:
                            if offset <= res[0]:
                                break
                            offset += len(line) + 1
                    tc.setPosition(offset + res[0], QTextCursor.MoveAnchor)
                    tc.setPosition(offset + res[1], QTextCursor.KeepAnchor)
                else:
                    tc.setPosition(res[0], QTextCursor.MoveAnchor)
                    tc.setPosition(res[1], QTextCursor.KeepAnchor)
                self.ui.log_textedit.setTextCursor(tc)
                self.ui.log_textedit.ensureCursorVisible()
                self.count += 1
                if self.count == len(res_list):
                    self.count = 0
            except IndexError:
                self.count = 0
        else:
            QMessageBox.warning(self, '错误提示', '找不到' + search_text)

    def search_prev(self):
        search_text = self.ui.search_lineEdit.text()
        main_text = self.ui.log_textedit.toPlainText()
        res_list = []
        flags = 0
        if self.ui.case_check.isChecked():
            if not self.ui.regex_check.isChecked():
                flags = flags | re.IGNORECASE
        if self.ui.regex_check.isChecked():
            try:
                pattern = re.compile(search_text, flags)
                if self.start_date and self.end_date:
                    for line in main_text.splitlines():
                        timestamp_str = line[:19]
                        timestamp = QDateTime.fromString(timestamp_str, "yyyy-MM-dd hh:mm:ss")
                        if self.start_date <= timestamp <= self.end_date:
                            for match in re.finditer(pattern, line[20:]):
                                res_list.append((match.start(), match.end()))
                else:
                    for match in re.finditer(pattern, main_text):
                        res_list.append(match.span())
            except re.error:
                QMessageBox.warning(self, '错误提示', '正则表达式错误。')
                return
        else:
            if self.start_date and self.end_date:
                for line in main_text.splitlines():
                    timestamp_str = line[:19]
                    timestamp = QDateTime.fromString(timestamp_str, "yyyy-MM-dd hh:mm:ss")
                    if self.start_date <= timestamp <= self.end_date:
                        for match in re.finditer(re.escape(search_text), line[20:], flags):
                            res_list.append((match.start(), match.end()))
            else:
                for match in re.finditer(re.escape(search_text), main_text, flags):
                    res_list.append(match.span())

        if res_list:
            try:
                self.count -= 1
                if self.count < 0:
                    self.count = len(res_list) - 1
                res = res_list[self.count]
                tc = self.ui.log_textedit.textCursor()
                if self.start_date and self.end_date:
                    offset = 0
                    for line in main_text.splitlines():
                        if self.start_date <= QDateTime.fromString(line[:19], "yyyy-MM-dd hh:mm:ss") <= self.end_date:
                            if offset <= res[0]:
                                break
                            offset += len(line) + 1
                    tc.setPosition(offset + res[0], QTextCursor.MoveAnchor)
                    tc.setPosition(offset + res[1], QTextCursor.KeepAnchor)
                else:
                    tc.setPosition(res[0], QTextCursor.MoveAnchor)
                    tc.setPosition(res[1], QTextCursor.KeepAnchor)
                self.ui.log_textedit.setTextCursor(tc)
                self.ui.log_textedit.ensureCursorVisible()
            except IndexError:
                self.count = 0
        else:
            QMessageBox.warning(self, '错误提示', '找不到' + search_text)

    def closeEvent(self, event):
        # 在窗口关闭时清理日志记录器资源
        self.logger.disabled = True
        self.logger.setLevel(logging.CRITICAL)
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.text_edit_handler)
        self.file_handler.close()
        self.ui.log_textedit.clear()
        event.accept()


#if __name__ == "__main__":
#    app = QApplication([])
#    widget = LoggerDialog()
#    widget.show()
#    sys.exit(app.exec_())
