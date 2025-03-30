import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

from ratte.config import Config

__all__ = ["DataPreprocessor"]


class LowPassFilter:
    def __init__(self, *, cutoff: float, fs: float, baseline: float, order: int = 3, eps=1e-6) -> None:
        self.baseline = baseline
        self.eps = eps
        nyq: float = 0.5 * fs  # Nyquist frequency
        normal_cutoff: float = cutoff / nyq
        # noinspection PyTupleAssignmentBalance
        self.b, self.a = butter(order, normal_cutoff, btype="low", analog=False)
        self.zi: np.ndarray = lfilter_zi(self.b, self.a)  # initial state

        # warm up the filter
        while abs(self.filter_sample(baseline) - baseline) > eps:
            pass

    def filter_sample(self, sample: float) -> float:
        y, self.zi = lfilter(self.b, self.a, [sample], zi=self.zi)  # process sample
        return y[0]

    def reset(self):
        self.zi = lfilter_zi(self.b, self.a)

        # warm up the filter
        while abs(self.filter_sample(self.baseline) - self.baseline) > self.eps:
            pass


class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.filters = {
            wsk_id: LowPassFilter(
                fs=config.control_rps,
                cutoff=wsk_config.lowpass_cutoff,
                baseline=wsk_config.lowpass_baseline,
            )
            for wsk_id, wsk_config in config.whiskers.items()
        }

    def preprocess_defl(self, defl: float, wsk_id: str) -> float:
        return self.filters[wsk_id].filter_sample(defl)

    def reset(self):
        for f in self.filters.values():
            f.reset()
