from src.base_fem import BaseFem


class Interpolation(BaseFem):
    def __init__(self, n, shape):
        super().__init__(n, shape)
        self.set_node_values()

    def set_node_values(self):
        f = self.function
        x = self.omega.x_coords
        y = self.omega.y_coords
        for i in range(len(x)):
            self.node_values.append(f(x[i], y[i]))
