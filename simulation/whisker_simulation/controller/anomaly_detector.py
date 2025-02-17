import numpy as np

from whisker_simulation.models import WorldState, Mode
from whisker_simulation.utils import get_monitor, get_logger

__all__ = ["AnomalyDetector"]

logger = get_logger(__file__)
monitor = get_monitor()


class AnomalyDetector:
    def __init__(self, *, total_v: float):
        self.total_v = total_v

    def run(
        self,
        *,
        state: WorldState,
        prev_state: WorldState,
        tip_w: np.ndarray,
        tip_w_prev: np.ndarray,
    ) -> tuple[Mode, str] | None:
        time_step = state.time - prev_state.time

        # assume that when the system has exited idle, steady body velocity has been reached
        body_dr = state.body_r_w - prev_state.body_r_w
        body_v = np.linalg.norm(body_dr) / time_step
        if (diff_p := abs(body_v / self.total_v - 1)) > 0.1:
            logger.warning(f"body_v={body_v:.3f}, total_velocity={self.total_v:.3f}")
            # return (
            #     Mode.FAILURE,
            #     f"Expected body velocity ({self.total_v:.2f}) "
            #     f"differs from actual ({body_v:.2f}) by {diff_p:.2%}",
            # )

        # now we are sure that the body has covered some distance
        # check whether the whisker is slipping (sliding backwards)
        # control might not be respected, so rely on the previous world state
        tip_dr = tip_w - tip_w_prev
        tip_v = np.linalg.norm(tip_dr) / time_step
        # the tip might be stuck, so ignore small movements
        if tip_v / self.total_v >= 0.2:
            if (angle := np.dot(body_dr, tip_dr)) < 0:
                return (
                    Mode.SLIPPING_BACKWARDS,
                    f"Angle between body and tip is {angle:.2f}",
                )

        # the whisker might be slipping forward, but that's alright
        # still, we want to detect this
        # the indicator is that the tip is moving faster than the body
        if (dv_p := tip_v / self.total_v - 1) > 0.5:
            logger.warning(
                f"Whisker is slipping forwards: tip velocity ({tip_v:.2f}) "
                f"is larger than total velocity ({self.total_v:.2f}) by {dv_p:.2%}"
            )

        # the whisker might become disengaged
        # the indicator is that the tip is moving faster than the body
        # and the deflection is vanishing
        # TODO
        return None
