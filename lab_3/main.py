import numpy as np
import matplotlib.pyplot as plt

from src.interpolation import Interpolation

xa, xb = 0, 1
ya, yb = 0, 1
shape = np.array([[xa, ya], [xa, yb], [xb, yb], [xb, ya]])


def print_errors():
    print("function:")
    for i in range(1, 9):
        inter = Interpolation(2**i, shape)
        print("n = ", 2**i, ", error = ", inter.error(is_grad=False))

    print("gradient:")
    for i in range(1, 9):
        inter = Interpolation(2**i, shape)
        print("n = ", 2**i, ", error = ", inter.error(is_grad=True))


def draw_errors():
    n = [2**x for x in range(1, 9)]
    err = [Interpolation(i, shape).error() for i in n]
    err_grad = [Interpolation(i, shape).error(True) for i in n]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    x, y = n, err
    ax1.plot(x, y)
    x, y = n, err_grad
    ax2.plot(x, y, 'r')

    ax1.set_title("func_error(num_of_nodes)")
    ax1.grid()
    ax1.set_xlabel("nodes")
    ax1.set_ylabel("error")

    ax2.set_title("grad_error(num_of_nodes)")
    ax2.grid()
    ax2.set_xlabel("nodes")
    ax2.set_ylabel("gradient error")

    plt.show()


if __name__ == "__main__":
    print_errors()
    # draw_errors()



