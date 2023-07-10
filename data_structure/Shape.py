

class Shape():

    axis = []

    def __init__(self, parameter):
        self.axis = parameter

    def get_axis(self):
        return self.axis

    def set_axis(self, axis_param):
        self.axis = axis_param
