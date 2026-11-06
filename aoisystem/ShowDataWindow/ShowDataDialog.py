# This Python file uses the following encoding: utf-8
#import pymysql
import logging
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QInputDialog, \
                            QLineEdit, QMessageBox, QApplication
from PyQt5.QtCore import Qt
from PyQt5 import uic


# 设置日志记录器
logging.basicConfig(filename='./Logger/aoi_system.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_data_from_mysql():
    # 连接 MySQL 数据库
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="123456",
        database="aoi_image"
    )
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM images")
    data = cursor.fetchall()
    cursor.close()
    mydb.close()
    return data


class ShowDataDialog(QDialog):
    def __init__(self):
        super(ShowDataDialog, self).__init__()
        self.ui = uic.loadUiType("./AOI/ShowDataWindow/ShowData.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowTitle("数据展示")

        self.ui.add_button.clicked.connect(self.add_item)
        self.ui.edit_button.clicked.connect(self.edit_item)
        self.ui.delete_button.clicked.connect(self.delete_item)

        self.populate_table()

    def populate_table(self):
        data = get_data_from_mysql()
        self.ui.table_widget.setRowCount(len(data))
        self.ui.table_widget.setColumnCount(len(data[0]))
        column_names = ["ID", "Name", "Szie", "Type"]
        self.ui.table_widget.setHorizontalHeaderLabels(column_names)

        for row, row_data in enumerate(data):
            for col, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.ui.table_widget.setItem(row, col, item)

    def add_item(self):
        # 在此处实现添加功能
        name, ok = QInputDialog().getText(self, "输入图片名称", "请输入图片名称：")
        if ok and name:
            type, ok = QInputDialog().getText(self, "输入图片类型", "请输入图片类型：")
            if ok and type:
                size_str, ok = QInputDialog().getText(self, "输入图片大小", "请输入图片大小：")
                if ok:
                    try:
                        size = int(size_str)
                        mydb = pymysql.connect(
                            host="localhost",
                            user="root",
                            password="123456",
                            database="aoi_image"
                        )
                        cursor = mydb.cursor()
                        sql = "INSERT INTO images (image_name, image_type, image_size) VALUES (%s, %s, %s)"
                        val = (name, type, size)
                        cursor.execute(sql, val)
                        mydb.commit()
                        cursor.close()
                        mydb.close()
                        logging.info(f'添加了图片：名称={name}，类型={type}，大小={size}')
                        self.populate_table()
                    except ValueError:
                        logging.warning('警告：图片大小必须是整数。')
                        QMessageBox.warning(self, "错误", "图片大小必须是整数。")

    def edit_item(self):
        # 在此处实现编辑功能
        selected_row = self.ui.table_widget.currentRow()
        if selected_row >= 0:
            id = int(self.ui.table_widget.item(selected_row, 0).text())
            name, ok = QInputDialog().getText(self, "编辑图片名称", "请输入新的图片名称：", QLineEdit.Normal, self.ui.table_widget.item(selected_row, 1).text())
            if ok and name:
                type, ok = QInputDialog().getText(self, "编辑图片类型", "请输入新的图片类型：", QLineEdit.Normal, self.ui.table_widget.item(selected_row, 3).text())
                if ok and type:
                    size_str, ok = QInputDialog().getText(self, "编辑图片大小", "请输入新的图片大小：", QLineEdit.Normal, self.ui.table_widget.item(selected_row, 2).text())
                    if ok:
                        try:
                            size = int(size_str)
                            mydb = pymysql.connect(
                                host="localhost",
                                user="root",
                                password="123456",
                                database="aoi_image"
                            )
                            cursor = mydb.cursor()
                            sql = "UPDATE images SET image_name = %s, image_type = %s, image_size = %s WHERE id = %s"
                            val = (name, type, size, id)
                            cursor.execute(sql, val)
                            mydb.commit()
                            cursor.close()
                            mydb.close()
                            logging.info(f'编辑了图片 ID={id}，新名称={name}，新类型={type}，新大小={size}')
                            self.populate_table()
                        except ValueError:
                            logging.warning('警告：图片大小必须是整数。')
                            QMessageBox.warning(self, "错误", "图片大小必须是整数。")

    def delete_item(self):
        # 在此处实现删除功能
        selected_row = self.ui.table_widget.currentRow()
        if selected_row >= 0:
            id = int(self.ui.table_widget.item(selected_row, 0).text())
            reply = QMessageBox.question(self, '确认删除', "您确定要删除此条记录吗？", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                mydb = pymysql.connect(
                    host="localhost",
                    user="root",
                    password="123456",
                    database="aoi_image"
                )
                cursor = mydb.cursor()
                sql = "DELETE FROM images WHERE id = %s"
                val = (id,)
                cursor.execute(sql, val)
                mydb.commit()
                cursor.close()
                mydb.close()
                now = logging.Formatter.converter(None)
                logging.info(f'{now} - 删除了图片 ID={id}')
                self.populate_table()


if __name__ == "__main__":
    app = QApplication([])
    widget = ShowDataDialog()
    widget.show()
    app.exec_()
