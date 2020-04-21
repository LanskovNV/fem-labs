import numpy as np

from interpolation import Interpolation


if __name__ == "__main__":
    num_of_nodes = 2
    x0, x1 = 0, 1
    y0, y1 = 0, 1
    shape = np.array([[x0, y0], [x0, y1], [x1, y1], [x0, y1]])

    inter = 0
    print("function:")
    for i in range(8):
        inter = Interpolation(num_of_nodes, shape)
        print("n = ", num_of_nodes, ", error = ", inter.error(is_grad=False))
        num_of_nodes *= 2

    num_of_nodes = 2
    print("gradient:")
    for i in range(8):
        inter = Interpolation(num_of_nodes, shape)
        print("n = ", num_of_nodes, ", error = ", inter.error(is_grad=True))
        num_of_nodes *= 2

    inter.draw()
