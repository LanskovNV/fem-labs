import matplotlib.pyplot as plt
import numpy as np


def draw(omega, function):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = omega.x_coords
    y = omega.y_coords
    func = np.vectorize(function)
    z = func(x, y)

    ax.plot_trisurf(x, y, z)

    plt.show()


def middle_val(trc, a, b, x, y):
    xa, xb = trc[a][0], trc[b][0]
    ya, yb = trc[a][1], trc[b][1]
    za, zb = trc[a][2], trc[b][2]

    if xb - xa == 0:
        return za + (zb - za) * (y - ya) / (yb - ya)
    else:
        return za + (zb - za) * (x - xa) / (xb - xa)


def middle_val_grad(trc):
    m = np.column_stack((trc[:, 0], trc[:, 1], np.ones(3)))
    r = np.array([
        trc[0][2],
        trc[1][2],
        trc[2][2]
    ])
    a = np.linalg.solve(m, r)
    return a[0] + a[1]


def area(trc):
    s = (trc[1, 0] - trc[0, 0]) * (trc[2, 1] - trc[0, 1]) - \
        (trc[1, 1] - trc[0, 1]) * (trc[2, 0] - trc[0, 0])
    s = abs(s) / 2
    return s