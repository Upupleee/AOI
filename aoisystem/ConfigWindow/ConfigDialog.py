import sys
import logging
import configparser
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QDoubleValidator
from PyQt5 import uic


# 设置日志记录器
logging.basicConfig(filename='./Logger/aoi_system.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ConfigDialog(QDialog):
    def __init__(self):
        super(ConfigDialog, self).__init__()
        self.ui = uic.loadUiType("./ConfigWindow/ConfigSet.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowTitle("参数设置")

        # 限定输入格式
        # 概要
        self.ui.lineEdit_BoardThickness.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_EtchValue.setValidator(QDoubleValidator(0, float('inf'), 3))

        # 设计规范
        self.ui.lineEdit_MinLineWidth.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_MinLineSpacing.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_MinCopperSlag.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_MinPinHole.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_LineTypeDefect.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_IndependentCopper.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_IndependentPinHole.setValidator(QDoubleValidator(0.001, 1000, 3))

        # 线宽
        self.ui.lineEdit_LinesMin.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_LinesMax.setValidator(QDoubleValidator(0.001, 1000, 3))

        # 焊盘
        self.ui.lineEdit_PadsMax.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_PadsMin.setValidator(QDoubleValidator(0.001, 1000, 3))

        # SMT
        self.ui.lineEdit_SMT_High.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_SMT_Width.setValidator(QDoubleValidator(0.001, 1000, 3))

        # 钻孔
        self.ui.lineEdit_hole_ring.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_hole_break.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_drilling_minimal.setValidator(QDoubleValidator(0.001, 1000, 3))
        self.ui.lineEdit_drilling_maximal.setValidator(QDoubleValidator(0.001, 1000, 3))

        # 铜面层
        self.ui.lineEdit_coppper_tolerance.setValidator(QDoubleValidator(0.001, 1000, 3))

        # 区域
        # Targets

        # 参数设置概要
        self.ui.comboBox_Template.currentIndexChanged.connect(self.saveConfig) # type: ignore
        self.ui.comboBox_Material.currentIndexChanged.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_BoardThickness.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_EtchValue.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.comboBox_BoardPolarity.currentIndexChanged.connect(self.saveConfig) # type: ignore
        self.ui.comboBox_Resolution.currentIndexChanged.connect(self.saveConfig) # type: ignore

        # 参数设置设计规范
        self.ui.lineEdit_MinLineWidth.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_MinLineSpacing.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.comboBox_MicShort.currentIndexChanged.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_MinCopperSlag.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_MinPinHole.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_LineTypeDefect.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.comboBox_Dishdown.currentIndexChanged.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_IndependentCopper.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_IndependentPinHole.editingFinished.connect(self.saveConfig) # type: ignore

        # 参数设置线宽
        self.ui.lineEdit_LinesMin.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_LinesMax.editingFinished.connect(self.saveConfig) # type: ignore

        # 参数设置焊盘
        self.ui.lineEdit_PadsMax.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_PadsMin.editingFinished.connect(self.saveConfig) # type: ignore

        # 参数设置SMT
        self.ui.lineEdit_SMT_High.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_SMT_Width.editingFinished.connect(self.saveConfig) # type: ignore

        # 参数设置钻孔
        self.ui.lineEdit_hole_ring.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_hole_break.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_drilling_minimal.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_drilling_maximal.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.radioButton_hole_ring.toggled.connect(self.saveConfig) # type: ignore
        self.ui.radioButton_hole_break.toggled.connect(self.saveConfig) # type: ignore
        self.ui.radioButton_hole_disappears.toggled.connect(self.saveConfig) # type: ignore
        self.ui.radioButton_no_report.toggled.connect(self.saveConfig) # type: ignore

        # 参数设置铜面层
        self.ui.lineEdit_coppper_tolerance.editingFinished.connect(self.saveConfig) # type: ignore
        self.ui.checkBox_copper.toggled.connect(self.saveConfig) # type: ignore
        self.ui.lineEdit_elelctric_formation.editingFinished.connect(self.saveConfig) # type: ignore

        # 参数设置区域

        # 参数设置Targets

        # 在初始化时加载配置
        self.loadConfig()

    def saveConfig(self):
        logging.info('保存配置操作')
        config = configparser.ConfigParser()
        config['tab_outline'] = {
                                 'TemplateValue': self.ui.comboBox_Template.currentText(),
                                 'MaterialValue': self.ui.comboBox_Material.currentText(),
                                 'BoardThicknessValue': self.ui.lineEdit_BoardThickness.text(),
                                 'EtchValue': self.ui.lineEdit_EtchValue.text(),
                                 'BoardPolarityValue': self.ui.comboBox_BoardPolarity.currentText(),
                                 'ResolutionValue': self.ui.comboBox_Resolution.currentText(),
                                 }
        config['tab_design'] = {
                                 'MinLineWidthValue': self.ui.lineEdit_MinLineWidth.text(),
                                 'MinLineSpacingValue': self.ui.lineEdit_MinLineSpacing.text(),
                                 'MicShortValue': self.ui.comboBox_MicShort.currentText(),                                
                                 'MinCopperSlagValue': self.ui.lineEdit_MinCopperSlag.text(),
                                 'MinPinHoleValue': self.ui.lineEdit_MinPinHole.text(),
                                 'LineTypeDefectValue': self.ui.lineEdit_LineTypeDefect.text(),
                                 'DishdownValue': self.ui.comboBox_Dishdown.currentText(),
                                 'IndependentCopperValue': self.ui.lineEdit_IndependentCopper.text(),
                                 'IndependentPinHoleValue': self.ui.lineEdit_IndependentPinHole.text(),
                                 'checkBox_LineTypeDefectValue': str(self.ui.checkBox_LineTypeDefect.isChecked()),
                                 'checkBox_IndependentCopperValue': str(self.ui.checkBox_IndependentCopper.isChecked()),
                                 'checkBox_IndependentPinHoleValue': str(self.ui.checkBox_IndependentPinHole.isChecked()),
        }
        config['tab_line'] = {
                             'LinesMinValue': self.ui.lineEdit_LinesMin.text(),
                             'LinesMaxValue': self.ui.lineEdit_LinesMax.text(),
                             }
        config['tab_pads'] = {
                             'PadsMinValue': self.ui.lineEdit_PadsMin.text(),
                             'PadsMaxValue': self.ui.lineEdit_PadsMax.text(),
                             }
        config['tab_smt'] = {
                             'SMT_HighValue': self.ui.lineEdit_SMT_High.text(),
                             'SMT_WidthValue': self.ui.lineEdit_SMT_Width.text(),
                             }
        config['tab_hole'] = {
                             'hole_ringValue': self.ui.lineEdit_hole_ring.text(),
                             'hole_breakValue': self.ui.lineEdit_hole_break.text(),
                             'drilling_minimalValue': self.ui.lineEdit_drilling_minimal.text(),
                             'drilling_maximalValue': self.ui.lineEdit_drilling_maximal.text(),
                             'radioButton_hole_ringValue': str(self.ui.radioButton_hole_ring.isChecked()),
                             'radioButton_hole_breakValue': str(self.ui.radioButton_hole_break.isChecked()),
                             'radioButton_hole_disappearsValue': str(self.ui.radioButton_hole_disappears.isChecked()),
                             'radioButton_no_reportValue': str(self.ui.radioButton_no_report.isChecked()),
                             }
        config['tab_copper'] = {
                             'coppper_toleranceValue': self.ui.lineEdit_coppper_tolerance.text(),
                             'checkBox_copperValue': str(self.ui.checkBox_copper.isChecked()),
                             'elelctric_formationValue': self.ui.lineEdit_elelctric_formation.text(),

        }
        with open('./ConfigWindow/config.ini', 'w') as configfile:
            config.write(configfile)

    def loadConfig(self):
        config = configparser.ConfigParser()
        if config.read('./ConfigWindow/config.ini'):  # 检查配置文件是否存在并读取
            if 'tab_outline' in config:
                selected_index_Template = self.ui.comboBox_Template.findText(config['tab_outline'].get('TemplateValue', '选项 1'))
                self.ui.comboBox_Template.setCurrentIndex(selected_index_Template)
                selected_index_Material = self.ui.comboBox_Material.findText(config['tab_outline'].get('MaterialValue', '选项 1'))
                self.ui.comboBox_Material.setCurrentIndex(selected_index_Material)
                self.ui.lineEdit_BoardThickness.setText(config['tab_outline'].get('BoardThicknessValue', ''))
                self.ui.lineEdit_EtchValue.setText(config['tab_outline'].get('EtchValue', ''))
                selected_index_BoardPolarity = self.ui.comboBox_BoardPolarity.findText(config['tab_outline'].get('BoardPolarityValue', '选项 1'))
                self.ui.comboBox_BoardPolarity.setCurrentIndex(selected_index_BoardPolarity)
                selected_index_Resolution = self.ui.comboBox_Resolution.findText(config['tab_outline'].get('ResolutionValue', '选项 1'))
                self.ui.comboBox_Resolution.setCurrentIndex(selected_index_Resolution)

            if 'tab_design' in config:
                self.ui.lineEdit_MinLineWidth.setText(config['tab_design'].get('MinLineWidthValue', ''))
                self.ui.lineEdit_MinLineSpacing.setText(config['tab_design'].get('MinLineSpacingValue', ''))
                selected_index_MicShort = self.ui.comboBox_MicShort.findText(config['tab_design'].get('MicShortValue', '选项 1'))
                self.ui.comboBox_MicShort.setCurrentIndex(selected_index_MicShort)
                self.ui.lineEdit_MinCopperSlag.setText(config['tab_design'].get('MinCopperSlagValue', ''))
                self.ui.lineEdit_MinPinHole.setText(config['tab_design'].get('MinPinHoleValue', ''))
                self.ui.lineEdit_LineTypeDefect.setText(config['tab_design'].get('LineTypeDefectValue', ''))
                selected_index_Dishdown = self.ui.comboBox_Dishdown.findText(config['tab_design'].get('DishdownValue', '选项 1'))
                self.ui.comboBox_Dishdown.setCurrentIndex(selected_index_Dishdown)
                self.ui.lineEdit_IndependentCopper.setText(config['tab_design'].get('IndependentCopperValue', ''))
                self.ui.lineEdit_IndependentPinHole.setText(config['tab_design'].get('IndependentPinHoleValue', ''))
                self.ui.checkBox_LineTypeDefect.setChecked(config.getboolean('tab_design', 'checkBox_LineTypeDefectValue'))
                self.ui.checkBox_IndependentCopper.setChecked(config.getboolean('tab_design', 'checkBox_IndependentCopperValue'))
                self.ui.checkBox_IndependentPinHole.setChecked(config.getboolean('tab_design', 'checkBox_IndependentPinHoleValue'))

            if 'tab_line' in config:
                self.ui.lineEdit_LinesMin.setText(config['tab_line'].get('LinesMinValue', ''))
                self.ui.lineEdit_LinesMax.setText(config['tab_line'].get('LinesMaxValue', ''))

            if 'tab_pads' in config:
                self.ui.lineEdit_PadsMin.setText(config['tab_pads'].get('PadsMinValue', ''))
                self.ui.lineEdit_PadsMax.setText(config['tab_pads'].get('PadsMaxValue', ''))

            if 'tab_smt' in config:
                self.ui.lineEdit_SMT_High.setText(config['tab_smt'].get('SMT_HighValue', ''))
                self.ui.lineEdit_SMT_Width.setText(config['tab_smt'].get('SMT_WidthValue', ''))

            if 'tab_hole' in config:
                self.ui.lineEdit_hole_ring.setText(config['tab_hole'].get('hole_ringValue', ''))
                self.ui.lineEdit_hole_break.setText(config['tab_hole'].get('hole_breakValue', ''))
                self.ui.lineEdit_drilling_minimal.setText(config['tab_hole'].get('drilling_minimalValue', ''))
                self.ui.lineEdit_drilling_maximal.setText(config['tab_hole'].get('drilling_maximalValue', ''))
                self.ui.radioButton_hole_ring.setChecked(config.getboolean('tab_hole', 'radioButton_hole_ringValue'))
                self.ui.radioButton_hole_break.setChecked(config.getboolean('tab_hole', 'radioButton_hole_breakValue'))
                self.ui.radioButton_hole_disappears.setChecked(config.getboolean('tab_hole', 'radioButton_hole_disappearsValue'))
                self.ui.radioButton_no_report.setChecked(config.getboolean('tab_hole', 'radioButton_no_reportValue'))

            if 'tab_copper' in config:
                self.ui.lineEdit_coppper_tolerance.setText(config['tab_copper'].get('coppper_toleranceValue', ''))
                self.ui.checkBox_copper.setChecked(config.getboolean('tab_copper', 'checkBox_copperValue'))
                self.ui.lineEdit_elelctric_formation.setText(config['tab_copper'].get('elelctric_formationValue', ''))


#if __name__ == "__main__":
#    app = QApplication([])
#    widget = ConfigDialog()
#    widget.show()
#    sys.exit(app.exec_())
