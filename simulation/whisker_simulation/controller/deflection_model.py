import numpy as np

__all__ = ["DeflectionModel"]


class DeflectionModel:
    @staticmethod
    def get_x(an):  # X
        poly = -1.41e18 * an**5 - 8.617e14 * an**4 - 9.859e10 * an**3 + 2.063e08 * an**2 + 5192 * an - 0.1542
        return (100 - poly) / 1000

    @staticmethod
    def get_y(an):  # Y
        poly = -1.389e19 * an**5 - 1.317e16 * an**4 - 4.362e12 * an**3 - 6.456e08 * an**2 - 2.075e05 * an - 2.149
        return -poly / 1000

    @staticmethod
    def get_position(an):
        x, y = DeflectionModel.get_x(an), DeflectionModel.get_y(an)
        return np.array([x, y])
