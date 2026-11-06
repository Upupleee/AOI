#ryanCheckAgain 
import sys
import yaml
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit
from PyQt5.QtCore import QSettings, pyqtSignal

class SystemSettingDialog(QDialog):
    close_thread_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUiType("./SystemSettingWindow/SystemSettingWindow.ui")[0]()
        self.ui.setupUi(self)

        self.acs_config_path = "./Acs/config_1104.yaml"
        self.monit_ini_path = "./SystemSettingWindow/motion_settings.ini"

        ##modified 0522 by Jingshu: change camera ccf path
        # self.ccf_path = "/home/aoi/Documents/T_LA_CM_16K05A_00_R_flatfield_0226.ccf"
        self.ccf_path = "/home/aoi/Documents/best_0429.ccf"
        ## close

        # 连接信号槽，当文本框内容改变时保存设置
        self.ui.lineEdit_x_speed.textChanged.connect(lambda: self.save_settings("x_speed"))
        self.ui.lineEdit_x_acceleration.textChanged.connect(lambda: self.save_settings("x_acceleration"))
        self.ui.lineEdit_x_distance.textChanged.connect(lambda: self.save_settings("x_distance"))
        self.ui.lineEdit_x_absolute_position.textChanged.connect(lambda: self.save_settings("x_absolute_position"))
        self.ui.lineEdit_x_relative_position.textChanged.connect(lambda: self.save_settings("x_relative_position"))
        self.ui.lineEdit_y_speed.textChanged.connect(lambda: self.save_settings("y_speed"))
        self.ui.lineEdit_y_acceleration.textChanged.connect(lambda: self.save_settings("y_acceleration"))
        self.ui.lineEdit_y_distance.textChanged.connect(lambda: self.save_settings("y_distance"))
        self.ui.lineEdit_y_absolute_position.textChanged.connect(lambda: self.save_settings("y_absolute_position"))
        self.ui.lineEdit_y_relative_position.textChanged.connect(lambda: self.save_settings("y_relative_position"))
        self.ui.lineEdit_z_speed.textChanged.connect(lambda: self.save_settings("z_speed"))
        self.ui.lineEdit_z_acceleration.textChanged.connect(lambda: self.save_settings("z_acceleration"))
        self.ui.lineEdit_z_distance.textChanged.connect(lambda: self.save_settings("z_distance"))
        self.ui.lineEdit_z_absolute_position.textChanged.connect(lambda: self.save_settings("z_absolute_position"))
        self.ui.lineEdit_z_relative_position.textChanged.connect(lambda: self.save_settings("z_relative_position"))
        self.ui.lineEdit_gain.textChanged.connect(lambda: self.save_settings("camrea_gain"))
        self.ui.lineEdit_times.textChanged.connect(lambda: self.save_settings("camrea_times"))
        self.ui.lineEdit_light.textChanged.connect(lambda: self.save_settings("light"))


        self.ui.spinBox_flag.valueChanged.connect(lambda:self.save_settings("flag"))
        self.ui.spinBox_slice.valueChanged.connect(lambda:self.save_settings("slice"))
        self.ui.spinBox_thres.valueChanged.connect(lambda:self.save_settings("thres"))
        self.ui.spinBox_row.valueChanged.connect(lambda:self.save_settings("row"))
        self.ui.spinBox_col.valueChanged.connect(lambda:self.save_settings("col"))
        self.ui.spinBox_overlap_samp.valueChanged.connect(lambda:self.save_settings("overlap_samp"))
        self.ui.spinBox_overlap_temp.valueChanged.connect(lambda:self.save_settings("overlap_temp"))
        self.ui.checkBox_Autosplit.stateChanged.connect(self.save_settings_checkbox)
        self.ui.checkBox_SaveImage.stateChanged.connect(self.save_settings_saveimage)

        # self.ui.lineEdit_crop_height.textChanged.connect(lambda: self.save_ccf_settings("crop_height"))

        # 运动轨迹参数设置
        self.ui.default_x_velocity_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("default_x_velocity"))
        self.ui.default_x_acceleration_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("default_x_acceleration"))
        self.ui.default_x_position_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("default_x_position"))
        self.ui.default_y_velocity_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("default_y_velocity"))
        self.ui.default_y_acceleration_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("default_y_acceleration"))
        self.ui.default_y_position_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("default_y_position"))

        

        # 板宽板高和横向移动距离设置
#        self.ui.board_width_lineEdit.textChanged.connect(self.value_changed)
        self.ui.board_width_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("board_width"))
        self.ui.board_height_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("board_height"))
        # self.ui.movement_distance_lineEdit.textChanged.connect(lambda: self.save_yaml_settings("movement_distance"))

        self.ui.confirm_button.clicked.connect(self.confirm_button_clicked)
        
        # 加载设置
        self.load_settings()

        self.load_yaml_settings()
        
        self.load_ccf_settings()

        

    def value_changed(self):
        board_height = self.ui.board_height_lineEdit.text().strip()
        crop_height = 129.706*(board_height+12.883)
        self.ui.lineEdit_crop_height.setText(str(crop_height))
        self.save_yaml_settings("board_height")
        # self.save_ccf_settings("crop_height")


    def confirm_button_clicked(self):
#        self.save_ccf_settings("crop_height")
        self.close_thread_signal.emit()
        self.close()

    
    # def load_ccf_settings(self):
    #     with open(self.ccf_path, "r") as f:
    #         for line in f:
    #             line = line.strip()  # 去除行首尾的空白字符
    #             if line.startswith('Crop Height' + '='):
    #                 parameter_value = line.split('=')[1]
    #                 break
    #     self.ui.lineEdit_crop_height.setText(parameter_value)

    # def save_ccf_settings(self, key):
    #     if key == "crop_height":
    #         new_value = self.ui.lineEdit_crop_height.text()
    #     lines = []
    #     with open(self.ccf_path, 'r') as f:
    #         lines = f.readlines()
    #     with open(self.ccf_path, "w") as f:
    #         for line in lines:
    #             line = line.strip()  # 去除行首尾的空白字符
    #             if line.startswith('Crop Height' + '='):
    #                 f.write(f"Crop Height={new_value}\n")
    #             else:
    #                 f.write(line + '\n')
                    # break

    def load_yaml_settings(self):
        with open(self.acs_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        axis_x = self.config.get("Axis_X", {})
        self.ui.default_x_velocity_lineEdit.setText(str(axis_x.get("default_velocity")))
        self.ui.default_x_acceleration_lineEdit.setText(str(axis_x.get("default_acceleration")))
        self.ui.default_x_position_lineEdit.setText(str(axis_x.get("initial_position")))
#        self.ui.default_x_back_velocity_lineEdit.setText(str(axis_x.get("back_movement_velocity")))
        axis_y = self.config.get("Axis_Y", {})
        self.ui.default_y_velocity_lineEdit.setText(str(axis_y.get("default_velocity")))
        self.ui.default_y_acceleration_lineEdit.setText(str(axis_y.get("default_acceleration")))
        self.ui.default_y_position_lineEdit.setText(str(axis_y.get("initial_position")))
        board = self.config.get("Board", {})
        self.ui.board_width_lineEdit.setText(str(board.get("width")))
        self.ui.board_height_lineEdit.setText(str(board.get("height")))
        # self.ui.movement_distance_lineEdit.setText(str(board.get("movement_distance")))

    
    def save_yaml_settings(self, key):
        if key == "default_x_velocity":
            self.config["Axis_X"]["default_velocity"] = float(self.ui.default_x_velocity_lineEdit.text().strip())
        elif key == "default_x_acceleration":
            self.config["Axis_X"]["default_acceleration"] = float(self.ui.default_x_acceleration_lineEdit.text().strip())
        elif key == "default_x_position":
            self.config["Axis_X"]["initial_position"] = float(self.ui.default_x_position_lineEdit.text().strip())
        elif key == "default_y_velocity":
            self.config["Axis_Y"]["default_velocity"] = float(self.ui.default_y_velocity_lineEdit.text().strip())
        elif key == "default_y_acceleration":
            self.config["Axis_Y"]["default_acceleration"] = float(self.ui.default_y_acceleration_lineEdit.text().strip())
        elif key == "default_y_position":
            self.config["Axis_Y"]["initial_position"] = float(self.ui.default_y_position_lineEdit.text().strip())
        elif key == "board_width":
            self.config["Board"]["width"] = float(self.ui.board_width_lineEdit.text().strip())
        elif key == "board_height":
            #ryanCheckAgain 
            self.config["Board"]["height"] = float(self.ui.board_height_lineEdit.text().strip())
            # self.config["Board"]["board_height"] = float(self.ui.board_height_lineEdit.text())
            crop_height = 129.706*(float(self.ui.board_height_lineEdit.text().strip())+12.883)
            self.ui.lineEdit_crop_height.setText(str(crop_height))
            # self.save_ccf_settings("crop_height")
        # elif key == "movement_distance":
        #     self.config["movement_distance"] = float(self.ui.movement_distance_lineEdit.text())
        with open(self.acs_config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)



    def load_settings(self):
        settings = QSettings(self.monit_ini_path, QSettings.IniFormat)
        self.ui.lineEdit_x_speed.setText(settings.value("x_speed", ""))
        self.ui.lineEdit_x_acceleration.setText(settings.value("x_acceleration", ""))
        self.ui.lineEdit_x_distance.setText(settings.value("x_distance", ""))
        self.ui.lineEdit_x_absolute_position.setText(settings.value("x_absolute_position", ""))
        self.ui.lineEdit_x_relative_position.setText(settings.value("x_relative_position", ""))
        self.ui.lineEdit_y_speed.setText(settings.value("y_speed", ""))
        self.ui.lineEdit_y_acceleration.setText(settings.value("y_acceleration", ""))
        self.ui.lineEdit_y_distance.setText(settings.value("y_distance", ""))
        self.ui.lineEdit_y_absolute_position.setText(settings.value("y_absolute_position", ""))
        self.ui.lineEdit_y_relative_position.setText(settings.value("y_relative_position", ""))
        self.ui.lineEdit_z_speed.setText(settings.value("z_speed", ""))
        self.ui.lineEdit_z_acceleration.setText(settings.value("z_acceleration", ""))
        self.ui.lineEdit_z_distance.setText(settings.value("z_distance", ""))
        self.ui.lineEdit_z_absolute_position.setText(settings.value("z_absolute_position", ""))
        self.ui.lineEdit_z_relative_position.setText(settings.value("z_relative_position", ""))
        self.ui.lineEdit_gain.setText(settings.value("camrea_gain", ""))
        self.ui.lineEdit_times.setText(settings.value("camrea_times", ""))
        self.ui.lineEdit_light.setText(settings.value("light", ""))

        self.ui.spinBox_flag.setValue(settings.value("flag",10,type=int))
        self.ui.spinBox_slice.setValue(settings.value("slice",720,type=int))
        self.ui.spinBox_thres.setValue(settings.value("thres",0.05,type=float))
        self.ui.spinBox_row.setValue(settings.value("row",3,type=int))
        self.ui.spinBox_col.setValue(settings.value("col",2,type=int))
        self.ui.spinBox_overlap_samp.setValue(settings.value("overlap_samp",type=int))
        self.ui.spinBox_overlap_temp.setValue(settings.value("overlap_temp",type=int))
        self.ui.checkBox_Autosplit.setChecked(settings.value("Auto_split",False,type=bool))
        self.ui.checkBox_SaveImage.setChecked(settings.value("SaveImage",False,type=bool))

    def save_settings(self, key):
        settings = QSettings(self.monit_ini_path, QSettings.IniFormat)
        if key == "x_speed":
            settings.setValue(key, self.ui.lineEdit_x_speed.text())
        elif key == "x_acceleration":
            settings.setValue(key, self.ui.lineEdit_x_acceleration.text())
        elif key == "x_distance":
            settings.setValue(key, self.ui.lineEdit_x_distance.text())
        elif key == "x_absolute_position":
            settings.setValue(key, self.ui.lineEdit_x_absolute_position.text())
        elif key == "x_relative_position":
            settings.setValue(key, self.ui.lineEdit_x_relative_position.text())
        elif key == "y_speed":
            settings.setValue(key, self.ui.lineEdit_y_speed.text())
        elif key == "y_acceleration":
            settings.setValue(key, self.ui.lineEdit_y_acceleration.text())
        elif key == "y_distance":
            settings.setValue(key, self.ui.lineEdit_y_distance.text())
        elif key == "y_absolute_position":
            settings.setValue(key, self.ui.lineEdit_y_absolute_position.text())
        elif key == "y_relative_position":
            settings.setValue(key, self.ui.lineEdit_y_relative_position.text())
        elif key == "z_speed":
            settings.setValue(key, self.ui.lineEdit_z_speed.text())
        elif key == "z_acceleration":
            settings.setValue(key, self.ui.lineEdit_z_acceleration.text())
        elif key == "z_distance":
            settings.setValue(key, self.ui.lineEdit_z_distance.text())
        elif key == "z_absolute_position":
            settings.setValue(key, self.ui.lineEdit_z_absolute_position.text())
        elif key == "z_relative_position":
            settings.setValue(key, self.ui.lineEdit_z_relative_position.text())
        elif key == "camrea_gain":
            settings.setValue(key, self.ui.lineEdit_gain.text())
        elif key == "camrea_times":
            settings.setValue(key, self.ui.lineEdit_times.text())
        elif key == "light":
            settings.setValue(key, self.ui.lineEdit_light.text())
        elif key == "flag":
            settings.setValue(key, self.ui.spinBox_flag.value())
        elif key == "slice":
            settings.setValue(key, self.ui.spinBox_slice.value())
        elif key == "thres":
            settings.setValue(key, self.ui.spinBox_thres.value())
        elif key == "overlap_temp":
            settings.setValue(key, self.ui.spinBox_overlap_temp.value())
        elif key == "overlap_samp":
            settings.setValue(key, self.ui.spinBox_overlap_samp.value())
        elif key == "row":
            settings.setValue(key, self.ui.spinBox_row.value())
        elif key == "col":
            settings.setValue(key, self.ui.spinBox_col.value())


    def save_settings_checkbox(self):
        settings = QSettings(self.monit_ini_path, QSettings.IniFormat)
        auto_split = self.ui.checkBox_Autosplit.isChecked()
        settings.setValue("Auto_split", auto_split)

    def save_settings_saveimage(self):
        settings = QSettings(self.monit_ini_path, QSettings.IniFormat)
        save_image = self.ui.checkBox_SaveImage.isChecked()
        settings.setValue("SaveImage", save_image)
      
