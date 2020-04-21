import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d

from rectangle import Rectangle


class Interpolation:
    def __init__(self, func, n, shape=np.array([[0, 0], [0, 1], [1, 1], [1, 0]])):
        self.interpolant = 0
        self.shape = shape
        self.function = func
        self.omega = Rectangle(self.shape)
        self.omega.triangulate(n)
        self.__interpolate()

    def __interpolate(self):
        func = np.vectorize(self.function)
        self.interpolant = func(self.omega.x_coords, self.omega.y_coords)

    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = self.omega.x_coords
        y = self.omega.y_coords
        z = self.interpolant

        ax.plot_trisurf(x, y, z)

        plt.show()
