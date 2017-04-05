import numpy as np
from nn_layer import *

class NN_model:
    def __init__(self, layers, optimizers):
        self.layers = layers
        self.optimizers = optimizers

    def step(self, input_data, answer):
        for layer in self.layers: 
            outputs = layer.cal_output(input_data)
            input_data = outputs[1]
        self.layers[-1].cal_back_grad(None, None, answer, self.optimizers[-1].regularization_coefficient, np.shape(input_data)[1])
        for i in reversed(range(len(self.layers)-1)): self.layers[i].cal_back_grad(self.layers[i+1].back_grad, self.layers[i+1].W, None, self.optimizers[i].regularization_coefficient, np.shape(input_data)[1])
        for i, layer in enumerate(self.layers): layer.step(self.optimizers[i])
    
    def eval(self, input_data, answer):
        for layer in self.layers:
            outputs = layer.cal_output(input_data)
            input_data = outputs[1]
        accuracy = 1.0 - np.float64((np.mean(np.logical_xor(np.round(input_data), answer))))
        return accuracy
       
    def predict(self, input_data):
        for layer in self.layers:
            outputs = layer.cal_output(input_data)
            input_data = outputs[1]
        return input_data
