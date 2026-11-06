"""AOI 扫描流程状态机。

定义扫描检测全流程的状态枚举与状态机,纯 Python 实现,不依赖 PyQt5。
负责状态合法性校验、状态转换通知,不持有任何业务逻辑。
"""
import logging
from enum import Enum
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


class ScanState(Enum):
    """扫描流程状态枚举。"""

    IDLE = "idle"                          # 空闲,等待用户操作
    MATERIAL_SELECTED = "material_selected"  # 已选料号,等待开始
    CAMERA_STARTING = "camera_starting"    # 正在启动相机
    PLC_READY = "plc_ready"                # PLC software_ready 已置位
    ACS_MOVING = "acs_moving"              # ACS 运动中,正在采图
    IMAGE_QUEUEING = "image_queueing"      # 图片入队,等待检测
    DETECTING = "detecting"                # 正在模板匹配+缺陷检测
    ROUND_COMPLETE = "round_complete"      # 本轮检测完成
    AB_CYCLE = "ab_cycle"                  # AB 面切换
    FINISHED = "finished"                  # 全部完成
    ERROR = "error"                        # 错误状态


class ScanStateMachine:
    """扫描流程状态机。

    维护当前扫描状态,校验状态转换合法性,并在转换后通知监听器。
    所有状态转换必须显式调用 :meth:`transition`,非法转换抛出 ``ValueError``。
    """

    def __init__(self):
        # 当前状态,初始为 IDLE
        self._state: ScanState = ScanState.IDLE
        # 状态变化监听器列表,callback 形式: callback(state: ScanState) -> None
        self._listeners: List[Callable[[ScanState], None]] = []

    @property
    def current_state(self) -> ScanState:
        """当前状态。"""
        return self._state

    def _get_valid_transitions(self) -> Dict[ScanState, List[ScanState]]:
        """返回合法状态转换映射表。

        Returns:
            dict: key 为源状态,value 为可转换的目标状态列表。
        """
        # 任何状态均可转 ERROR,ERROR 仅可转 IDLE,
        # FINISHED 仅可转 IDLE(重置)。
        return {
            ScanState.IDLE: [
                ScanState.MATERIAL_SELECTED,
                ScanState.ERROR,
            ],
            ScanState.MATERIAL_SELECTED: [
                ScanState.CAMERA_STARTING,
                ScanState.ERROR,
            ],
            ScanState.CAMERA_STARTING: [
                ScanState.PLC_READY,
                ScanState.ERROR,
            ],
            ScanState.PLC_READY: [
                ScanState.ACS_MOVING,
                ScanState.ERROR,
            ],
            ScanState.ACS_MOVING: [
                ScanState.IMAGE_QUEUEING,
                ScanState.ERROR,
            ],
            ScanState.IMAGE_QUEUEING: [
                ScanState.DETECTING,
                ScanState.ERROR,
            ],
            ScanState.DETECTING: [
                ScanState.ROUND_COMPLETE,
                ScanState.ERROR,
            ],
            ScanState.ROUND_COMPLETE: [
                ScanState.AB_CYCLE,
                ScanState.FINISHED,
                ScanState.ERROR,
            ],
            ScanState.AB_CYCLE: [
                ScanState.ACS_MOVING,
                ScanState.FINISHED,
                ScanState.ERROR,
            ],
            ScanState.FINISHED: [
                ScanState.IDLE,
            ],
            ScanState.ERROR: [
                ScanState.IDLE,
            ],
        }

    def can_transition(self, new_state: ScanState) -> bool:
        """检查从当前状态是否可以转换到 ``new_state``。

        Args:
            new_state: 目标状态。

        Returns:
            bool: 允许转换返回 True,否则 False。
        """
        valid_targets = self._get_valid_transitions().get(self._state, [])
        return new_state in valid_targets

    def transition(self, new_state: ScanState):
        """执行状态转换。

        校验转换合法性,非法时抛出 ``ValueError``。
        转换成功后通知所有监听器。

        Args:
            new_state: 目标状态。

        Raises:
            ValueError: 当前状态不允许转换到 ``new_state``。
        """
        if not self.can_transition(new_state):
            raise ValueError(
                f"非法状态转换: {self._state.value} -> {new_state.value}"
            )
        old_state = self._state
        self._state = new_state
        logger.info(
            "状态转换: %s -> %s", old_state.value, new_state.value
        )
        # 通知所有监听器
        for callback in list(self._listeners):
            try:
                callback(new_state)
            except Exception:
                # 监听器异常不应影响状态机本身
                logger.exception(
                    "状态监听器回调异常,目标状态: %s", new_state.value
                )

    def add_listener(self, callback: Callable[[ScanState], None]):
        """添加状态变化监听器。

        Args:
            callback: 回调函数,签名为 ``callback(state: ScanState)``。
        """
        if callback in self._listeners:
            logger.debug("监听器已存在,跳过添加: %r", callback)
            return
        self._listeners.append(callback)
        logger.debug("已添加状态监听器: %r", callback)

    def remove_listener(self, callback: Callable[[ScanState], None]):
        """移除状态变化监听器。

        Args:
            callback: 待移除的回调函数。
        """
        if callback in self._listeners:
            self._listeners.remove(callback)
            logger.debug("已移除状态监听器: %r", callback)
        else:
            logger.debug("未找到待移除的监听器: %r", callback)

    def reset(self):
        """重置状态机到 IDLE。

        若当前为 ERROR,走合法路径 ERROR -> IDLE;
        若当前为 FINISHED,走合法路径 FINISHED -> IDLE;
        其他情况下直接将内部状态置回 IDLE,并记录日志。
        """
        if self._state == ScanState.IDLE:
            logger.debug("状态机已处于 IDLE,无需重置")
            return
        if self.can_transition(ScanState.IDLE):
            self.transition(ScanState.IDLE)
        else:
            # 兜底:强行回到 IDLE(用于异常恢复场景)
            old_state = self._state
            self._state = ScanState.IDLE
            logger.warning(
                "状态机强制重置: %s -> idle(非标准转换路径)",
                old_state.value,
            )

    def is_idle(self) -> bool:
        """当前是否处于 IDLE。"""
        return self._state == ScanState.IDLE

    def is_running(self) -> bool:
        """当前是否正在运动或检测中。

        覆盖 CAMERA_STARTING / PLC_READY / ACS_MOVING / IMAGE_QUEUEING / DETECTING。
        """
        return self._state in (
            ScanState.CAMERA_STARTING,
            ScanState.PLC_READY,
            ScanState.ACS_MOVING,
            ScanState.IMAGE_QUEUEING,
            ScanState.DETECTING,
        )

    def is_error(self) -> bool:
        """当前是否处于 ERROR。"""
        return self._state == ScanState.ERROR
