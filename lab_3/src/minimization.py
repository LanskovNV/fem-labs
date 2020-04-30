import numpy as np
from src.base_fem import BaseFem


class Minimization(BaseFem):
    def __init__(self, n, shape):
        super().__init__(n, shape)
        self.set_node_values()

    def set_node_values(self):
        rank = len(self.omega.x_coords)
        x = self.omega.x_coords
        y = self.omega.y_coords
        A = np.zeros((rank, rank))
        R = np.zeros(rank)

        for i in range(rank):
            elems_i = self.get_elems(i)
            shapes_i = self.shapes(elems_i, i)

            # right part
            value = 0
            for si in shapes_i:
                value += si(x(i), y(i)) * self.function(x(i), y(i))
            for ei in elems_i:
                R[i] += self.integrate(ei, value, i)

            # matrix row
            for j in range(rank):
                elems_j = self.get_elems(j)
                shapes_j = self.shapes(elems_j, j)
                value = 0
                for si in shapes_i:
                    for sj in shapes_j:
                        value += si(x(i), y(i)) * sj(x(i), y(i))

                for ei in elems_i:
                    A[i, j] += self.integrate(ei, value, i)

        self.node_values = np.linalg.solve(A, R)
