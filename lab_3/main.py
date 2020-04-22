import numpy as np
import matplotlib.pyplot as plt
from math import log2

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

    f = np.vectorize(log2)
    def f1(x): return -2*x - 3
    def f2(x): return - x - 2
    f1 = np.vectorize(f1)
    f2 = np.vectorize(f2)
    z = range(1, 9)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    x, y = n, err
    ax1.plot(f(x), f(y), label='func_err')
    ax1.plot(z, f1(z), 'ro', label='y = -2x - 3')
    x, y = n, err_grad
    ax2.plot(f(x), f(y), label='grad_err')
    ax2.plot(z, f2(z), 'ro', label='y = -x - 2')

    ax1.set_title("func_error(num_of_nodes)")
    ax1.set_xlabel("log(nodes)")
    ax1.set_ylabel("log(error)")
    ax1.grid()
    ax1.legend()

    ax2.set_title("grad_error(num_of_nodes)")
    ax2.set_xlabel("log(nodes)")
    ax2.set_ylabel("log(gradient error)")
    ax2.grid()
    ax2.legend()

    plt.savefig("pic/Figure_3.png")
    # plt.show()



if __name__ == "__main__":
    # print_errors()
    draw_errors()



