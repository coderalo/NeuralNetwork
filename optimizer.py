import numpy as np

class Optimizer:
    def __init__(self, parameters):
        self.optimizer = parameters['optimizer']
        if self.optimizer == "adam":
            self.step_size = parameters['step_size']
            self.exponential_decay_rate1 = parameters['exponential_decay_rate1']
            self.exponential_decay_rate2 = parameters['exponential_decay_rate2']
            self.first_moment_power = 1.0
            self.second_moment_power = 1.0
            self.epsilon = parameters['epsilon']
        elif self.optimizer == "adadelta":
            self.decay_coefficient = parameters['decay_coefficient']
            self.epsilon = parameters['epsilon']
        else:
            self.learning_rate = parameters['learning_rate']
        self.regularization_coefficient = parameters['regularization_coefficient']
