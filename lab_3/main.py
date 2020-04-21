import numpy as np
from math import sin, pi

from interpolation import Interpolation


if __name__ == "__main__":
    num_of_nodes = 20
    def function(x, y): return sin( - x * y)
    shape = np.array([[-pi, -1], [-pi, 1], [pi, 1], [pi, -1]])

    inter = Interpolation(function, num_of_nodes, shape)

    inter.draw()

