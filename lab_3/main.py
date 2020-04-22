import numpy as np
import matplotlib.pyplot as plt

from interpolation import Interpolation


if __name__ == "__main__":
    num_of_nodes = 2
    xa, xb = 0, 1
    ya, yb = 0, 1
    shape = np.array([[xa, ya], [xa, yb], [xb, yb], [xa, yb]])

    # Interpolation(num_of_nodes, shape).omega.draw_net()

    print("function:")
    for i in range(9):
        inter = Interpolation(num_of_nodes, shape)
        print("n = ", num_of_nodes, ", error = ", inter.error(is_grad=False))
        num_of_nodes *= 2

    num_of_nodes = 2
    print("gradient:")
    for i in range(9):
        inter = Interpolation(num_of_nodes, shape)
        print("n = ", num_of_nodes, ", error = ", inter.error(is_grad=True))
        num_of_nodes *= 2

