from functools import partial

import numpy as np

from whisker_simulation.models import ControllerState, WhiskerData
from whisker_simulation.utils import get_logger

__all__ = ["AnomalyDetector"]


class AnomalyDetector:
    def __init__(self, *, controller):
        from whisker_simulation.controller import Controller

        self.logger = get_logger(__file__, log_level=controller.config.log_level)
        self.ctrl: Controller = controller
        self.log_anomalies = self.ctrl.config.log_anomalies
        self.total_v = self.ctrl.config.body.total_v
        self.control_dt = self.ctrl.control_dt

        self.has_abnormal_velocity = False
        self.abnormal_velocity_start_time = None

        self.is_slipping = False
        self.slip_start_time = None

        self.is_disengaged = False
        self.disengaged_start_time = None

    def __call__(self) -> tuple[ControllerState, str] | None:
        m = self.ctrl.motion

        # check whether the control period is respected
        assert m.dt <= self.control_dt * 1.1, f"Time step {m.dt} is larger than the control period {self.control_dt}"

        checks = [
            self.detect_abnormal_velocity,
            (
                self.ctrl.active_wsk_id
                and partial(self.detect_abnormal_deflection, wsk=m.data.whiskers[self.ctrl.active_wsk_id])
            ),
            (self.ctrl.active_wsk_id and self.detect_slippage),
            (self.ctrl.active_wsk_id and self.detect_detached_tip),
        ]
        for check in checks:
            if check and (result := check()):
                return result
        return None

    def detect_abnormal_deflection(self, wsk: WhiskerData) -> tuple[ControllerState, str] | None:
        if wsk.orientation * self.ctrl.tgt_orient == -1 and abs(wsk.defl) > wsk.config.defl_threshold * 2:
            return ControllerState.FAILURE, "Deflection sign changed"

    def detect_abnormal_velocity(self) -> tuple[ControllerState, str] | None:
        # assume that when the system has exited exploring state
        # and steady body velocity has been reached
        time, m = self.ctrl.data.time, self.ctrl.motion
        if abs(m.body.v / self.total_v - 1) > 0.25:
            if not self.has_abnormal_velocity:
                self.has_abnormal_velocity = True
                self.abnormal_velocity_start_time = time
        elif self.has_abnormal_velocity:
            if self.log_anomalies:
                self.logger.debug(f"Abnormal velocity duration: {time - self.abnormal_velocity_start_time:.3f}")
            self.has_abnormal_velocity = False
            self.abnormal_velocity_start_time = None

    def detect_slippage(self) -> tuple[ControllerState, str] | None:
        # now we are sure that the body has covered some distance
        # check whether the whisker is slipping (sliding backwards)
        # control might not be respected, so rely on the previous world data
        # the tip might be stuck, so ignore small movements
        time, m = self.ctrl.data.time, self.ctrl.motion
        body, wsk = m.body, m.for_whisker(self.ctrl.active_wsk_id)
        if wsk.tip_drift_v / body.v > 0.5 and (angle := np.dot(wsk.tip_drift_v_w, body.v_w)) < -1e3:
            return (
                ControllerState.FAILURE,
                f"Slipping backwards: angle between body and tip is {angle:.2f}",
            )

        # the whisker might be slipping forward and that's alright -- still, we want to detect this
        # the indicator is that the tip is drifting
        if wsk.tip_drift_v / self.total_v > 0.5:
            if not self.is_slipping:
                self.is_slipping = True
                self.slip_start_time = time
        elif self.is_slipping:
            if time - self.slip_start_time > m.dt * 1.5:
                if self.log_anomalies:
                    self.logger.debug(f"Whisker slip duration: {time - self.slip_start_time:.3f}")
            self.is_slipping = False
            self.slip_start_time = None

    def detect_detached_tip(self) -> tuple[ControllerState, str] | None:
        # check whether the tip is detached from the body
        time = self.ctrl.data.time
        if not self.ctrl.wsk.is_deflected:
            if not self.is_disengaged:
                self.is_disengaged = True
                self.disengaged_start_time = time

            if time - self.disengaged_start_time > self.ctrl.wsk.config.disengaged_duration_threshold:
                return (
                    ControllerState.DISENGAGED,
                    f"No deflection for {time - self.disengaged_start_time:.3f}s",
                )
        elif self.is_disengaged:
            if self.log_anomalies:
                self.logger.debug(f"Whisker disengaged duration: {time - self.disengaged_start_time:.3f}")
            self.is_disengaged = False
            self.disengaged_start_time = None
