import numpy as np

__all__ = ["DeflectionModelRight", "DeflectionModelLeft"]


class DeflectionModelRight:
    @classmethod
    def __x(cls, an: float | np.ndarray):
        poly = -1.41e18 * an**5 - 8.617e14 * an**4 - 9.859e10 * an**3 + 2.063e08 * an**2 + 5192 * an - 0.1542
        return (100 - poly) / 1000

    @classmethod
    def __y(cls, an: float | np.ndarray):
        poly = -1.389e19 * an**5 - 1.317e16 * an**4 - 4.362e12 * an**3 - 6.456e08 * an**2 - 2.075e05 * an - 2.149
        return -poly / 1000

    @classmethod
    def __r(cls, an: float | np.ndarray) -> np.ndarray:
        sign = np.sign(an)
        an = -np.abs(an)
        x = cls.__x(an)
        y = -sign * cls.__y(-np.abs(an))
        return np.column_stack((x, y))

    @classmethod
    def r(cls, an: float | np.ndarray) -> np.ndarray:
        length = np.linalg.norm(cls.__r(0))
        r = cls.__r(-an)  # the model was originally defined for negative angles
        np.clip(r[:, 0], 1e-6, length, out=r[:, 0])
        np.clip(r[:, 1], -length, length, out=r[:, 1])
        r /= max(np.linalg.norm(r) / length, 1)
        if np.isscalar(an):
            return r[0]
        return r

    def __call__(self, an: float | np.ndarray) -> np.ndarray:
        return self.r(an)


class DeflectionModelLeft(DeflectionModelRight):
    @classmethod
    def r(cls, an: float | np.ndarray) -> np.ndarray:
        r = super().r(an)
        if np.isscalar(an):
            r[1] *= -1
            return r
        r[:, 1] *= -1
        return r
