import numpy as np
import matplotlib.pyplot as plt

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

        def int_middle(trc, a, b,  x, y):
            ff = self.function
            x0, x1 = trc[a][0], trc[b][0]
            y0, y1 = trc[a][1], trc[b][1]
            z0, z1 = ff(x0, y0), ff(x1, y1)
            dx, dy = x1 - x0, y1 - y0

            if is_grad:
                res = 0
                if x1 - x0 != 0:
                    res += (z1-z0) / (x1-x0)
                if y1 - y0 != 0:
                    res += (z1-z0) / (y1-y0)
                return res
            else:
                if dx == 0:
                    return z0 + (z1 - z0) * (y - y0) / dy
                else:
                    return z0 + (z1 - z0) * (x - x0) / dx

        def middle(trc, a, b):
            if is_grad:
                f = self.grad
            else:
                f = self.function
            x = (trc[a][0] - trc[b][0]) / 2
            y = (trc[a][1] - trc[b][1]) / 2

            return abs(int_middle(trc, a, b, x, y) - f(x, y))

        def int_triang(tr):
            trc = get_coords(tr)
            f01 = middle(trc, 0, 1)
            f12 = middle(trc, 1, 2)
            f02 = middle(trc, 0, 2)
            return area(trc) / 3 * (f01 * f12 * f02)

        ans = 0
        for _ in self.omega.triangles.triangles:
            ans += (int_triang(_))**2

        return ans

