import numpy as np

from whisker_simulation.models import ControllerState
from whisker_simulation.utils import get_logger, get_monitor

__all__ = ["AnomalyDetector"]

logger = get_logger(__file__)
monitor = get_monitor()


class AnomalyDetector:
    def __init__(self, *, controller):
        from whisker_simulation.controller import Controller

        self.ctrl: Controller = controller
        self.total_v = controller.total_v
        self.control_dt = controller.control_dt

        self.has_abnormal_velocity = False
        self.abnormal_velocity_start_time = None

        self.is_slipping = False
        self.slip_start_time = None

        self.is_disengaged = False
        self.disengaged_start_time = None

    def run(
        self,
        *,
        tip_w: np.ndarray,
        tip_w_prev: np.ndarray,
        is_deflected: bool,
        ignore_disengaged: bool = False,
    ) -> tuple[ControllerState, str] | None:
        data, prev_data = self.ctrl.data, self.ctrl.prev_data

        # check whether the control period is respected
        time_step = data.time - prev_data.time
        assert time_step <= self.control_dt * 1.1, (
            f"Time step {time_step} is larger than the control period {self.control_dt}"
        )

        # assume that when the system has exited exploring state
        # and steady body velocity has been reached
        body_dr = data.body_r_w - prev_data.body_r_w
        body_v = np.linalg.norm(body_dr) / time_step
        if abs(body_v / self.total_v - 1) > 0.25:
            if not self.has_abnormal_velocity:
                self.has_abnormal_velocity = True
                self.abnormal_velocity_start_time = data.time
        elif self.has_abnormal_velocity:
            logger.debug(f"Abnormal velocity duration: {data.time - self.abnormal_velocity_start_time:.3f}")
            self.has_abnormal_velocity = False
            self.abnormal_velocity_start_time = None

        # now we are sure that the body has covered some distance
        # check whether the whisker is slipping (sliding backwards)
        # control might not be respected, so rely on the previous world data
        tip_dr_w = tip_w - tip_w_prev
        tip_v_w = np.linalg.norm(tip_dr_w) / time_step
        # TODO: use tip_v_s instead of tip_v_w (account for the body rotation)
        # the tip might be stuck, so ignore small movements
        if tip_v_w / self.total_v >= 0.2:
            if (angle := np.dot(body_dr, tip_dr_w)) < -1e3:
                return (
                    ControllerState.FAILURE,
                    f"Slipping backwards: angle between body and tip is {angle:.2f}",
                )

        # the whisker might be slipping forward, but that's alright
        # still, we want to detect this
        # the indicator is that the tip is moving faster than the body
        if tip_v_w / self.total_v > 1.5:
            if not self.is_slipping:
                self.is_slipping = True
                self.slip_start_time = data.time
        elif self.is_slipping:
            logger.debug(f"Whisker slip duration: {data.time - self.slip_start_time:.3f}")
            self.is_slipping = False
            self.slip_start_time = None

        if not is_deflected:
            if not self.is_disengaged:
                self.is_disengaged = True
                self.disengaged_start_time = data.time

            if (
                not ignore_disengaged
                and data.time - self.disengaged_start_time > self.ctrl.disengaged_duration_threshold
            ):
                return (
                    ControllerState.DISENGAGED,
                    f"No deflection for {data.time - self.disengaged_start_time:.3f}s",
                )
        elif self.is_disengaged:
            logger.debug(f"Whisker disengaged duration: {data.time - self.disengaged_start_time:.3f}")
            self.is_disengaged = False
            self.disengaged_start_time = None

        return None
