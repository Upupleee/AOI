"""扫描流程状态机单元测试。

覆盖 ScanStateMachine 的状态校验、合法/非法转换、
监听器通知、状态查询以及完整业务流程。
"""
import pytest

from core.state_machine import ScanState, ScanStateMachine


class TestScanStateMachine:
    """扫描状态机测试。"""

    def test_initial_state(self):
        """初始状态应为 IDLE。"""
        sm = ScanStateMachine()
        assert sm.current_state is ScanState.IDLE
        assert sm.is_idle() is True

    def test_valid_transition(self):
        """IDLE -> MATERIAL_SELECTED 是合法转换。"""
        sm = ScanStateMachine()
        assert sm.can_transition(ScanState.MATERIAL_SELECTED) is True
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert sm.current_state is ScanState.MATERIAL_SELECTED

    def test_invalid_transition(self):
        """IDLE -> DETECTING 非法,应抛出 ValueError。"""
        sm = ScanStateMachine()
        assert sm.can_transition(ScanState.DETECTING) is False
        with pytest.raises(ValueError):
            sm.transition(ScanState.DETECTING)
        # 状态未变更
        assert sm.current_state is ScanState.IDLE

    def test_can_transition(self):
        """can_transition 在不同当前状态下返回正确结果。"""
        sm = ScanStateMachine()
        # IDLE 状态:合法目标
        assert sm.can_transition(ScanState.MATERIAL_SELECTED) is True
        assert sm.can_transition(ScanState.ERROR) is True
        # IDLE 状态:非法目标
        assert sm.can_transition(ScanState.DETECTING) is False
        assert sm.can_transition(ScanState.ACS_MOVING) is False
        assert sm.can_transition(ScanState.FINISHED) is False

        # 进入 MATERIAL_SELECTED 后
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert sm.can_transition(ScanState.CAMERA_STARTING) is True
        assert sm.can_transition(ScanState.ERROR) is True
        # 不能从 MATERIAL_SELECTED 直接跳到 DETECTING
        assert sm.can_transition(ScanState.DETECTING) is False

    def test_listener_notification(self):
        """添加监听器后,转换应触发回调,传入新状态。"""
        sm = ScanStateMachine()
        received = []
        sm.add_listener(lambda state: received.append(state))
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert received == [ScanState.MATERIAL_SELECTED]
        # 二次转换应再次触发
        sm.transition(ScanState.CAMERA_STARTING)
        assert received == [ScanState.MATERIAL_SELECTED, ScanState.CAMERA_STARTING]

    def test_listener_multiple(self):
        """多个监听器均应被通知。"""
        sm = ScanStateMachine()
        received_a = []
        received_b = []
        sm.add_listener(lambda s: received_a.append(s))
        sm.add_listener(lambda s: received_b.append(s))
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert received_a == [ScanState.MATERIAL_SELECTED]
        assert received_b == [ScanState.MATERIAL_SELECTED]

    def test_listener_remove(self):
        """移除监听器后不再被通知。"""
        sm = ScanStateMachine()
        received = []

        def cb(state):
            received.append(state)

        sm.add_listener(cb)
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert received == [ScanState.MATERIAL_SELECTED]

        sm.remove_listener(cb)
        sm.transition(ScanState.CAMERA_STARTING)
        # 移除后未再收到通知
        assert received == [ScanState.MATERIAL_SELECTED]

    def test_listener_duplicate_add(self):
        """重复添加同一监听器只应注册一次。"""
        sm = ScanStateMachine()
        received = []

        def cb(state):
            received.append(state)

        sm.add_listener(cb)
        sm.add_listener(cb)  # 重复
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert received == [ScanState.MATERIAL_SELECTED]

    def test_listener_exception_isolated(self):
        """监听器抛异常不应影响状态机本身与其他监听器。"""
        sm = ScanStateMachine()
        received = []

        def bad_cb(state):
            raise RuntimeError("监听器异常")

        def good_cb(state):
            received.append(state)

        sm.add_listener(bad_cb)
        sm.add_listener(good_cb)
        # 不应抛异常
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert sm.current_state is ScanState.MATERIAL_SELECTED
        # 正常监听器仍被通知
        assert received == [ScanState.MATERIAL_SELECTED]

    def test_reset_from_finished(self):
        """从 FINISHED reset 应走合法路径回到 IDLE。"""
        sm = ScanStateMachine()
        # 推进到 FINISHED(走最短路径:IDLE -> ERROR -> IDLE 不通,
        # 这里直接走真实流程:IDLE -> MATERIAL_SELECTED -> ... -> FINISHED)
        sm.transition(ScanState.MATERIAL_SELECTED)
        sm.transition(ScanState.CAMERA_STARTING)
        sm.transition(ScanState.PLC_READY)
        sm.transition(ScanState.ACS_MOVING)
        sm.transition(ScanState.IMAGE_QUEUEING)
        sm.transition(ScanState.DETECTING)
        sm.transition(ScanState.ROUND_COMPLETE)
        sm.transition(ScanState.FINISHED)
        assert sm.current_state is ScanState.FINISHED
        sm.reset()
        assert sm.current_state is ScanState.IDLE

    def test_reset_from_error(self):
        """从 ERROR reset 应回到 IDLE。"""
        sm = ScanStateMachine()
        sm.transition(ScanState.ERROR)
        assert sm.is_error() is True
        sm.reset()
        assert sm.is_idle() is True

    def test_reset_when_idle(self):
        """IDLE 状态下 reset 不应有副作用。"""
        sm = ScanStateMachine()
        sm.reset()
        assert sm.current_state is ScanState.IDLE

    def test_reset_from_middle_state(self):
        """从中间状态 reset 应强行回到 IDLE(兜底路径)。"""
        sm = ScanStateMachine()
        sm.transition(ScanState.MATERIAL_SELECTED)
        sm.transition(ScanState.CAMERA_STARTING)
        # CAMERA_STARTING 不能直接转 IDLE,走兜底
        sm.reset()
        assert sm.current_state is ScanState.IDLE

    def test_is_idle(self):
        """is_idle 状态查询。"""
        sm = ScanStateMachine()
        assert sm.is_idle() is True
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert sm.is_idle() is False

    def test_is_running(self):
        """is_running 覆盖 CAMERA_STARTING/PLC_READY/ACS_MOVING/IMAGE_QUEUEING/DETECTING。"""
        sm = ScanStateMachine()
        # IDLE 不算 running
        assert sm.is_running() is False
        # 进入 MATERIAL_SELECTED 不算 running
        sm.transition(ScanState.MATERIAL_SELECTED)
        assert sm.is_running() is False
        # CAMERA_STARTING 算 running
        sm.transition(ScanState.CAMERA_STARTING)
        assert sm.is_running() is True
        sm.transition(ScanState.PLC_READY)
        assert sm.is_running() is True
        sm.transition(ScanState.ACS_MOVING)
        assert sm.is_running() is True
        sm.transition(ScanState.IMAGE_QUEUEING)
        assert sm.is_running() is True
        sm.transition(ScanState.DETECTING)
        assert sm.is_running() is True
        # ROUND_COMPLETE 之后不再 running
        sm.transition(ScanState.ROUND_COMPLETE)
        assert sm.is_running() is False

    def test_is_error(self):
        """is_error 状态查询。"""
        sm = ScanStateMachine()
        assert sm.is_error() is False
        sm.transition(ScanState.ERROR)
        assert sm.is_error() is True
        sm.transition(ScanState.IDLE)
        assert sm.is_error() is False

    def test_full_flow(self):
        """完整流程:IDLE -> MATERIAL_SELECTED -> ... -> FINISHED -> IDLE。"""
        sm = ScanStateMachine()
        sm.transition(ScanState.MATERIAL_SELECTED)
        sm.transition(ScanState.CAMERA_STARTING)
        sm.transition(ScanState.PLC_READY)
        sm.transition(ScanState.ACS_MOVING)
        sm.transition(ScanState.IMAGE_QUEUEING)
        sm.transition(ScanState.DETECTING)
        sm.transition(ScanState.ROUND_COMPLETE)
        sm.transition(ScanState.FINISHED)
        assert sm.current_state is ScanState.FINISHED
        sm.transition(ScanState.IDLE)
        assert sm.current_state is ScanState.IDLE

    def test_full_flow_with_ab_cycle(self):
        """AB 面循环流程:ROUND_COMPLETE -> AB_CYCLE -> ACS_MOVING -> ..."""
        sm = ScanStateMachine()
        sm.transition(ScanState.MATERIAL_SELECTED)
        sm.transition(ScanState.CAMERA_STARTING)
        sm.transition(ScanState.PLC_READY)
        sm.transition(ScanState.ACS_MOVING)
        sm.transition(ScanState.IMAGE_QUEUEING)
        sm.transition(ScanState.DETECTING)
        sm.transition(ScanState.ROUND_COMPLETE)
        # 进入 AB 面切换
        sm.transition(ScanState.AB_CYCLE)
        assert sm.current_state is ScanState.AB_CYCLE
        # AB_CYCLE 可回到 ACS_MOVING(继续检测 B 面)
        sm.transition(ScanState.ACS_MOVING)
        assert sm.current_state is ScanState.ACS_MOVING
        # 也可由 AB_CYCLE 直接到 FINISHED
        # (这里走到第二轮 ROUND_COMPLETE 后再 FINISHED)
        sm.transition(ScanState.IMAGE_QUEUEING)
        sm.transition(ScanState.DETECTING)
        sm.transition(ScanState.ROUND_COMPLETE)
        sm.transition(ScanState.FINISHED)
        assert sm.current_state is ScanState.FINISHED

    def test_error_recovery(self):
        """任意运行中状态都可以转到 ERROR,再回到 IDLE。"""
        # 进入运行状态
        for target_state in [
            ScanState.CAMERA_STARTING,
            ScanState.PLC_READY,
            ScanState.ACS_MOVING,
            ScanState.IMAGE_QUEUEING,
            ScanState.DETECTING,
        ]:
            sm = ScanStateMachine()
            sm.transition(ScanState.MATERIAL_SELECTED)
            # 推进到目标状态
            path = [
                ScanState.CAMERA_STARTING,
                ScanState.PLC_READY,
                ScanState.ACS_MOVING,
                ScanState.IMAGE_QUEUEING,
                ScanState.DETECTING,
            ]
            for s in path:
                sm.transition(s)
                if s is target_state:
                    break
            assert sm.current_state is target_state
            # 任意运行状态都可转 ERROR
            assert sm.can_transition(ScanState.ERROR) is True
            sm.transition(ScanState.ERROR)
            assert sm.is_error() is True
            # ERROR -> IDLE 恢复
            sm.transition(ScanState.IDLE)
            assert sm.is_idle() is True

    def test_error_recovery_from_round_complete(self):
        """ROUND_COMPLETE 也可进入 ERROR 再恢复。"""
        sm = ScanStateMachine()
        sm.transition(ScanState.MATERIAL_SELECTED)
        sm.transition(ScanState.CAMERA_STARTING)
        sm.transition(ScanState.PLC_READY)
        sm.transition(ScanState.ACS_MOVING)
        sm.transition(ScanState.IMAGE_QUEUEING)
        sm.transition(ScanState.DETECTING)
        sm.transition(ScanState.ROUND_COMPLETE)
        sm.transition(ScanState.ERROR)
        assert sm.is_error() is True
        sm.transition(ScanState.IDLE)
        assert sm.is_idle() is True

    def test_no_transition_to_same_state(self):
        """同状态转换(如 IDLE -> IDLE)应非法。"""
        sm = ScanStateMachine()
        assert sm.can_transition(ScanState.IDLE) is False
        with pytest.raises(ValueError):
            sm.transition(ScanState.IDLE)

    def test_finished_can_only_go_idle(self):
        """FINISHED 状态只能转 IDLE。"""
        sm = ScanStateMachine()
        # 推进到 FINISHED
        sm.transition(ScanState.MATERIAL_SELECTED)
        sm.transition(ScanState.CAMERA_STARTING)
        sm.transition(ScanState.PLC_READY)
        sm.transition(ScanState.ACS_MOVING)
        sm.transition(ScanState.IMAGE_QUEUEING)
        sm.transition(ScanState.DETECTING)
        sm.transition(ScanState.ROUND_COMPLETE)
        sm.transition(ScanState.FINISHED)
        # FINISHED 不能转 ERROR
        assert sm.can_transition(ScanState.ERROR) is False
        # FINISHED 只能转 IDLE
        assert sm.can_transition(ScanState.IDLE) is True
