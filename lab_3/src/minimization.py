import numpy as np
from src.base_fem import BaseFem
from src.source import area


class Minimization(BaseFem):
    def __init__(self, n, shape):
        super().__init__(n, shape)
        self.set_node_values()

    def get_elems(self, i):
        ans = []
        for _ in self.omega.triangles.triangles:
            if i in _:
                ans.append(_)
        return ans

    def get_coords_xy(self, tr):
        omega = self.omega
        trc = np.zeros((3, 2))
        trc[0][0], trc[0][1] = omega.x_coords[tr[0]], omega.y_coords[tr[0]]
        trc[1][0], trc[1][1] = omega.x_coords[tr[1]], omega.y_coords[tr[1]]
        trc[2][0], trc[2][1] = omega.x_coords[tr[2]], omega.y_coords[tr[2]]
        return trc

    def shapes(self, elements_i, i):
        shapes = []
        for tr in elements_i:
            trc = self.get_coords_xy(tr)
            m = np.column_stack((trc, np.ones(3)))
            r = np.zeros(3)

            for j in range(3):
                if tr[j] == i:
                    r[j] = 1

            a = np.linalg.solve(m, r)
            shapes.append(lambda x, y: a[0] * x + a[1] * y + a[2])
        return shapes

    def integrate(self, ei, value):
        trc = self.get_coords_xy(ei)
        return area(trc) / 3 * 2 * value

    def set_node_values(self):
        rank = len(self.omega.x_coords)
        x = self.omega.x_coords
        y = self.omega.y_coords
        A = np.zeros((rank, rank))
        R = np.zeros(rank)

        for i in range(rank):
            elements_i = self.get_elems(i)
            shapes_i = self.shapes(elements_i, i)

            # right part
            value = 0
            for si in shapes_i:
                value += si(x[i], y[i]) * self.function(x[i], y[i])
            for ei in elements_i:
                R[i] += self.integrate(ei, value)

            # matrix row
            for j in range(rank):
                elements_j = self.get_elems(j)
                shapes_j = self.shapes(elements_j, j)
                value = 0
                for si in shapes_i:
                    for sj in shapes_j:
                        value += si(x[i], y[i]) * sj(x[i], y[i])

                for ei in elements_i:
                    A[i, j] += self.integrate(ei, value)

        self.node_values = np.linalg.solve(A, R)
