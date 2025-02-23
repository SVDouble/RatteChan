import numpy as np

__all__ = ["DeflectionModel"]


class DeflectionModel:
    @classmethod
    def __x(cls, an: float):
        poly = -1.41e18 * an**5 - 8.617e14 * an**4 - 9.859e10 * an**3 + 2.063e08 * an**2 + 5192 * an - 0.1542
        return (100 - poly) / 1000

    @classmethod
    def __y(cls, an: float):
        poly = -1.389e19 * an**5 - 1.317e16 * an**4 - 4.362e12 * an**3 - 6.456e08 * an**2 - 2.075e05 * an - 2.149
        return -poly / 1000

    @classmethod
    def r(cls, an: float) -> np.ndarray:
        sign = np.sign(an)
        an = -np.abs(an)
        x = cls.__x(an)
        y = -sign * cls.__y(-np.abs(an))
        if np.isscalar(an):
            return np.array([x, y])
        return np.column_stack((x, y))

    def __call__(self, an) -> np.ndarray:
        return self.r(an)
