import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from rectangle import Rectangle


class Interpolation:
    def __init__(self, n, shape=np.array([[0, 0], [0, 1], [1, 1], [1, 0]])):
        # omega setup
        self.shape = shape
        self.omega = Rectangle(self.shape)
        self.omega.triangulate(n)

        # function set up manually
        self.function = lambda x, y: x*(1 - x)*y*(1 - y)
        self.grad = lambda x, y: ((1 - 2*x)*(y - y**2) + (1 - 2*y)*(x - x**2))

    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = self.omega.x_coords
        y = self.omega.y_coords
        func = np.vectorize(self.function)
        z = func(self.omega.x_coords, self.omega.y_coords)

        ax.plot_trisurf(x, y, z)

        plt.show()

    def error(self, is_grad=False):
        def get_coords(tr):
            trc = np.zeros((3, 2))
            trc[0][0], trc[0][1] = self.omega.x_coords[tr[0]], self.omega.y_coords[tr[0]]
            trc[1][0], trc[1][1] = self.omega.x_coords[tr[1]], self.omega.y_coords[tr[1]]
            trc[2][0], trc[2][1] = self.omega.x_coords[tr[2]], self.omega.y_coords[tr[2]]
            return trc

        def area(trc):
            s = (trc[1, 0] - trc[0, 0])*(trc[2, 1] - trc[0, 1]) - \
                (trc[1, 1] - trc[0, 1])*(trc[2, 0] - trc[0, 0])
            s = abs(s) / 2
            return s

        def middle_val(trc, a, b, x, y):
            xa, xb = trc[a][0], trc[b][0]
            ya, yb = trc[a][1], trc[b][1]
            za, zb = self.function(xa, ya), self.function(xb, yb)

            if xb - xa == 0:
                return za + (zb - za) * (y - ya) / (yb - ya)
            else:
                return za + (zb - za) * (x - xa) / (xb - xa)

        def middle(trc, a, b):
            x = (trc[b][0] + trc[a][0]) / 2
            y = (trc[b][1] + trc[a][1]) / 2
            if is_grad:
                pass
            else:
                mid_val = middle_val(trc, a, b, x, y)
                return (abs(mid_val - self.function(x, y))) ** 2

        def int_triang(tr):
            trc = get_coords(tr)
            f01 = middle(trc, 0, 1)
            f12 = middle(trc, 1, 2)
            f20 = middle(trc, 2, 0)
            return area(trc) / 3 * (f01 + f12 + f20)

        ans = 0
        for _ in self.omega.triangles.triangles:
            ans += int_triang(_)

        return sqrt(ans)

