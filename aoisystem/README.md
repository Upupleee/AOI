# AOI 自动光学检测系统

基于 PyQt5 的 PCB 自动光学检测（AOI）桌面应用，采用**分层架构 + 依赖注入 + MVVM** 模式，集成运动控制、相机采图、缺陷检测与模板匹配四大子系统。

## 架构设计

```
┌─────────────────────────────────────────────────┐
│                  UI 层 (PyQt5)                    │
│  LoginView / MainView (纯视图,只负责显示和事件转发)  │
│  MainViewModel (业务逻辑代理,桥接 View 与 Service)  │
├─────────────────────────────────────────────────┤
│               业务服务层 (纯 Python)               │
│  MaterialService / ScanService / DetectionService │
│  VacuumService / HardwareInitializationService    │
│  ScanStateMachine (扫描流程状态机)                 │
├──────────────────┬──────────────────────────────┤
│   硬件实现层       │       数据持久化层             │
│  ModbusPlcClient  │  SqliteMaterialRepository    │
│  SpiiPlusAcsComm  │  SqliteScanRecordRepository  │
│  ProcessCamera    │  DatabaseManager (SQLite)    │
│  SahiDetector     │  JsonToSqliteMigrator        │
│  LoFtrMatcher     │                              │
├──────────────────┴──────────────────────────────┤
│            核心抽象层 (ABC 接口 + 数据模型)          │
│  IPlcClient / IAcsComm / ICamera / IDetector      │
│  ITemplateMatcher / IMaterialRepository           │
│  Material / ScanRecord / DetectionParams          │
├─────────────────────────────────────────────────┤
│             依赖注入容器 (Composition Root)         │
│  Container: 组装所有依赖,延迟创建单例               │
└─────────────────────────────────────────────────┘
```

### 设计原则

- **依赖倒置 (DIP)**:业务层只依赖 `core.interfaces` 抽象接口,不依赖具体实现
- **控制反转 (IoC)**:`core/container.py` 集中组装依赖,UI 层不直接 new 实现类
- **单一职责 (SRP)**:每个 Service/Repository 只负责一个领域
- **关注点分离**:业务逻辑(纯 Python)与 UI(PyQt5)完全分离,可独立测试

## 目录结构

```
aoisystem/
├── main.py                      # 入口:组装依赖并启动
├── config.yaml                  # 全局配置
├── requirements.txt
├── core/                        # 核心层(纯 Python,不依赖 Qt)
│   ├── interfaces/              #   抽象接口 (ABC)
│   │   ├── i_plc_client.py      #     IPlcClient
│   │   ├── i_acs_comm.py        #     IAcsComm
│   │   ├── i_camera.py          #     ICamera
│   │   ├── i_detector.py        #     IDetector + DetectionResult
│   │   ├── i_template_matcher.py#     ITemplateMatcher + MatchResult
│   │   └── i_repository.py      #     IMaterialRepository / IScanRecordRepository
│   ├── models/                  #   数据模型 (dataclass)
│   │   ├── material.py          #     Material
│   │   ├── scan_record.py       #     ScanRecord + ScanStatus
│   │   └── detection_params.py  #     DetectionParams
│   ├── services/                #   业务服务层
│   │   ├── material_service.py  #     料号管理 + 图片切分
│   │   ├── scan_service.py      #     扫描记录管理
│   │   ├── detection_service.py #     检测流程编排
│   │   ├── vacuum_service.py    #     真空/风机/报警控制
│   │   └── hardware_init_service.py # 硬件初始化
│   ├── state_machine/           #   状态机
│   │   └── scan_state_machine.py#     ScanState + ScanStateMachine
│   └── container.py             #   依赖注入容器
├── hardware/                    # 硬件实现层(实现 core.interfaces)
│   ├── plc/modbus_plc_client.py #   ModbusPlcClient(IPlcClient)
│   ├── acs/spiiplus_acs_comm.py #   SpiiPlusAcsComm(IAcsComm)
│   ├── camera/process_camera.py #   ProcessCamera(ICamera)
│   └── detection/
│       ├── sahi_detector.py     #   SahiDetector(IDetector)
│       └── loftr_template_matcher.py # LoFtrTemplateMatcher(ITemplateMatcher)
├── data/                        # 数据持久化层
│   ├── database.py              #   DatabaseManager (SQLite)
│   ├── repositories/            #   Repository 实现
│   │   ├── material_repository.py
│   │   └── scan_record_repository.py
│   └── json_migration.py        #   JSON -> SQLite 迁移工具
├── ui/                          # UI 层 (PyQt5)
│   ├── viewmodels/
│   │   └── main_viewmodel.py    #   MainViewModel (业务逻辑代理)
│   └── views/
│       ├── login_view.py        #   LoginView
│       └── main_view.py         #   MainView (纯视图)
├── common/                      # 共享基础
│   ├── config.py                #   AppConfig + 全局 config 单例
│   └── utils.py                 #   通用工具函数
├── tests/                       # 单元测试
│   ├── conftest.py              #   pytest fixtures (Mock 依赖)
│   ├── test_config.py
│   ├── test_utils.py
│   ├── test_models.py           #   数据模型测试
│   ├── test_state_machine.py    #   状态机测试
│   └── test_container.py        #   DI 容器测试
├── AoiSystemWindow_test/        # 旧业务模块(渐进式迁移中)
├── PLC/                         # PLC 通信(旧,逐步迁移到 hardware/)
├── Acs/                         # ACS 运动控制脚本与 buffer
├── DefectDetection/             # 缺陷检测 (sahi vendored + 模型)
├── TemplateMatching/            # 模板匹配 (EfficientLoFTR vendored)
├── ConfigWindow/                # 检测参数配置对话框
├── SystemSettingWindow/         # 运动参数设置对话框
├── Logger/                      # 日志查看对话框
├── LogoutWindow/                # 注销对话框
├── ShowDataWindow/              # 数据展示对话框
├── InitializeWindow/            # 初始化等待窗口
└── icons/                       # 图标资源
```

## 依赖注入

所有依赖通过 `core/container.py` 的 `Container` 类集中组装:

```python
from core.container import container

# 仓储层(立即可用)
material_repo = container.material_repository
scan_repo = container.scan_record_repository

# 服务层(自动注入仓储依赖)
material_service = container.material_service
scan_service = container.scan_service
detection_service = container.detection_service
vacuum_service = container.vacuum_service

# 硬件层(需先调用 initialize_hardware)
if container.initialize_hardware():  # 连接 PLC + ACS
    plc = container.plc_client
    acs = container.acs_comm

# 状态机
sm = container.state_machine
sm.transition(ScanState.MATERIAL_SELECTED)

# 关闭
container.shutdown()
```

## 状态机

扫描流程由状态机管理,非法转换抛出 `ValueError`:

```python
from core.state_machine import ScanState, ScanStateMachine

sm = ScanStateMachine()
sm.current_state              # ScanState.IDLE
sm.transition(ScanState.MATERIAL_SELECTED)
sm.transition(ScanState.CAMERA_STARTING)
sm.transition(ScanState.PLC_READY)
sm.transition(ScanState.ACS_MOVING)
# ...
sm.add_listener(lambda state: print(f"状态变更: {state.name}"))
```

## 配置

所有硬编码值集中在 [config.yaml](config.yaml):

```yaml
hardware:
  plc: {ip: "20.0.0.200", port: 502}
  acs: {ip: "10.0.0.100", port: 701}
paths:
  database: "aoi_system.db"
  material_root: "material"
  # ...
```

通过 `config` 单例访问:

```python
from common.config import config
config.plc_ip                  # "20.0.0.200"
config.database_path           # "aoi_system.db"
config.coil('vacuum')          # 255
config.register('temperature') # 202
```

## 安装与运行

### 环境要求

- Python 3.10
- CUDA 12.4 + TensorRT 10.6（GPU 推理）
- PyQt5

### 安装

```bash
pip install -r requirements.txt
```

### 启动(新架构)

```bash
python main.py
```

### 启动(旧架构,兼容)

```bash
python LoginWindow.py
```

## 测试

```bash
python -m pytest tests/ -v
```

测试覆盖:
- 数据模型(Material / ScanRecord / DetectionParams)
- 状态机(ScanStateMachine 完整流程)
- DI 容器(延迟创建 / 硬件未初始化保护)
- 配置与工具函数

所有测试使用 Mock 依赖,不连接真实硬件,不依赖 PyQt5。

## 数据迁移

从旧 JSON 文件迁移到 SQLite:

```python
from data.json_migration import JsonToSqliteMigrator
from core.container import container

migrator = JsonToSqliteMigrator()
migrator.migrate_materials("material_data.json", container.material_repository)
migrator.migrate_scan_records("scan_records.json", container.scan_record_repository)
```

## 技术栈

| 领域 | 技术 |
|------|------|
| GUI 框架 | PyQt5 + MVVM |
| 运动控制 | SPiiPlusPython (ACS) + Modbus PLC |
| 相机采图 | QProcess 调用 external_trigger |
| 缺陷检测 | SAHI + Ultralytics + TensorRT |
| 模板匹配 | EfficientLoFTR + LightGlue |
| 数据持久化 | SQLite (sqlite3 标准库) |
| 依赖注入 | 自建 Container (Composition Root) |
| 状态机 | 自建 ScanStateMachine |
| 配置 | YAML + AppConfig 单例 |
| 日志 | logging 标准库 |
| 测试 | pytest + unittest.mock |
