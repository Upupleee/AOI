"""依赖注入容器单元测试。

不连接真实硬件,验证 Container 的单例特性、延迟创建、
硬件未初始化保护以及配置可用性。
"""
import pytest

from common.config import AppConfig, config
from core.container import Container, container
from core.state_machine import ScanStateMachine


class TestContainer:
    """依赖注入容器测试。"""

    def test_container_singleton(self):
        """container 模块级变量应为全局单例。"""
        # 多次导入应返回同一对象
        from core.container import container as c1
        from core.container import container as c2
        assert c1 is container
        assert c2 is container
        assert c1 is c2

    def test_state_machine_available(self):
        """container.state_machine 应为 ScanStateMachine 实例。"""
        sm = container.state_machine
        assert isinstance(sm, ScanStateMachine)
        # 多次访问返回同一实例
        assert container.state_machine is sm

    def test_app_config_available(self):
        """container.app_config 应为全局 config 对象。"""
        cfg = container.app_config
        assert isinstance(cfg, AppConfig)
        assert cfg is config

    def test_material_repository_lazy(self):
        """首次访问创建实例,第二次返回缓存。

        使用全新的 Container 实例,避免全局单例状态污染。
        """
        c = Container()
        # 初始状态:仓储未创建
        assert c._material_repo is None
        # 首次访问触发创建
        repo_first = c.material_repository
        assert repo_first is not None
        assert c._material_repo is repo_first
        # 第二次访问返回缓存(同一实例)
        repo_second = c.material_repository
        assert repo_second is repo_first

    def test_scan_record_repository_lazy(self):
        """扫描记录仓储延迟创建并缓存。"""
        c = Container()
        assert c._scan_repo is None
        repo_first = c.scan_record_repository
        assert repo_first is not None
        assert c._scan_repo is repo_first
        repo_second = c.scan_record_repository
        assert repo_second is repo_first

    def test_plc_not_initialized(self):
        """未调用 initialize_hardware 时访问 plc_client 抛出 RuntimeError。"""
        c = Container()
        assert c._plc_client is None
        with pytest.raises(RuntimeError):
            _ = c.plc_client

    def test_acs_not_initialized(self):
        """未调用 initialize_hardware 时访问 acs_comm 抛出 RuntimeError。"""
        c = Container()
        assert c._acs_comm is None
        with pytest.raises(RuntimeError):
            _ = c.acs_comm

    def test_plc_available_after_set(self):
        """手动注入 plc_client 后访问属性应返回注入的实例。"""
        c = Container()
        # 模拟硬件初始化成功后内部字段被赋值
        fake_plc = object()
        c._plc_client = fake_plc
        assert c.plc_client is fake_plc

    def test_state_machine_per_container_instance(self):
        """每个 Container 实例应有独立的状态机。"""
        c1 = Container()
        c2 = Container()
        assert c1.state_machine is not c2.state_machine
        assert isinstance(c1.state_machine, ScanStateMachine)
        assert isinstance(c2.state_machine, ScanStateMachine)

    def test_container_construction_does_not_touch_hardware(self):
        """构造 Container 不应触发任何硬件连接。"""
        c = Container()
        # 所有硬件字段均应为 None
        assert c._plc_client is None
        assert c._acs_comm is None
        assert c._camera is None
        assert c._detector is None
        assert c._template_matcher is None

    def test_shutdown_resets_state_machine(self):
        """shutdown 应将状态机重置回 IDLE。"""
        c = Container()
        # 推进状态机到非 IDLE
        from core.state_machine import ScanState
        c.state_machine.transition(ScanState.MATERIAL_SELECTED)
        assert c.state_machine.is_idle() is False
        # shutdown 不应抛异常,且重置状态机
        c.shutdown()
        assert c.state_machine.is_idle() is True
