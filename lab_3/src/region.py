from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


class Region(ABC):
    triangles = []
    x_coords = []
    y_coords = []
    shape = []

    def __init__(self, shape):
        self.shape = shape

    def triangulate(self, n):
        self._flatten(n)
        x = self.x_coords
        y = self.y_coords
        self.triangles = tri.Triangulation(x, y)

    # generate flatten coords of the net points
    # with specified discretization steps
    @abstractmethod
    def _flatten(self, n):
        pass

    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        shape = plt.Polygon(self.shape, color='g', alpha=0.3)
        ax.add_patch(shape)
        plt.plot()
        plt.show()

    def draw_net(self):
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(self.triangles, 'bo-')
        plt.title('triplot of Delaunay triangulation')
        plt.show()