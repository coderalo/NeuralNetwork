import numpy as np
from utils import *

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def print_info(info):
    time = datetime.now()
    print '[{:0>2}:{:0>2}:{:0>2}] {}'.format(time.hour, time.minute, time.second, info)

class NN_layer:
    def __init__(self, layer, optimizer):
        output_feature_size = layer['output_feature_size']
        input_feature_size = layer['input_feature_size']
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        #self.W = np.zeros((output_feature_size, input_feature_size), dtype=np.float64)
        #self.b = np.zeros((output_feature_size, 1), dtype=np.float64)
        limit = np.sqrt(6.0 / (input_feature_size + output_feature_size))
        self.W = np.random.uniform(-limit, limit, (output_feature_size, input_feature_size))
        self.b = np.zeros((output_feature_size, 1), dtype=np.float64)
        #self.fore_grad = np.zeros((output_feature_size, input_feature_size+1), dtype=np.float64)
        #self.back_grad = np.zeros((output_feature_size, 1), dtype=np.float64)
        self.activation_function = layer['activation_function']
        self.loss = 0.0
        if optimizer.optimizer == "adagrad":
            self.accumulated_gradient = np.zeros((output_feature_size, input_feature_size+1), dtype=np.float64)
        elif optimizer.optimizer == "adadelta":
            self.accumulated_gradient = np.zeros((output_feature_size, input_feature_size+1), dtype=np.float64)
            self.accumulated_updates = np.zeros((output_feature_size, input_feature_size+1), dtype=np.float64)
        elif optimizer.optimizer == "adam":
            self.biased_first_moment_estimate = np.zeros((output_feature_size, input_feature_size+1), dtype=np.float64)
            self.biased_second_raw_moment_estimate = np.zeros((output_feature_size, input_feature_size+1), dtype=np.float64)
            self.bias_corrected_first_moment_estimate = np.zeros((output_feature_size, input_feature_size+1), dtype=np.float64)
            self.bias_corrected_second_moment_estimiate = np.zeros((output_feature_size, input_feature_size+1), dtype=np.float64)    

    def cal_output(self, input_data):
        self.raw_output_data = np.matmul(self.W, input_data) + self.b
        if self.activation_function == "unit": self.output_data = np.copy(self.raw_output_data)
        elif self.activation_function == "tanh": self.output_data = np.tanh(self.raw_output_data)
        elif self.activation_function == "sigmoid": self.output_data = sigmoid(self.raw_output_data)
        elif self.activation_function == "relu": self.output_data = np.maximum(self.raw_output_data, 0)
        self.fore_grad = np.zeros((np.shape(input_data)[1], self.input_feature_size+1), dtype=np.float64)
        self.fore_grad[:, :-1] = np.transpose(input_data)
        self.fore_grad[:, -1] = 1.0
        return [self.raw_output_data, self.output_data]

    def cal_loss_grad(self, answer):
        return -np.sum(answer * np.log(self.output_data) + (1.0 - answer) * np.log(1.0 - self.output_data)), self.output_data - answer

    def cal_back_grad(self, delta, next_layer_weight, answer, regularization_coefficient, input_size):
        if next_layer_weight == None and delta == None: 
            self.loss, self.back_grad = self.cal_loss_grad(answer)
            accuracy = 1.0 - np.float64((np.mean(np.logical_xor(np.round(self.output_data), answer))))
            print_info("Loss: {}, Accuracy: {}".format(np.sqrt(self.loss / input_size), accuracy))
        else:
            if self.activation_function == "unit": output_derivative = np.copy(self.output_data)
            elif self.activation_function == "tanh": output_derivative = 1 - self.output_data ** 2
            elif self.activation_function == "sigmoid": output_derivative = self.output_data * (1.0 - self.output_data)
            elif self.activation_function == "relu": output_derivative = np.float32(np.abs(self.output_data))
            self.back_grad = output_derivative * np.matmul(np.transpose(next_layer_weight), delta)
        self.grad = np.matmul(self.back_grad,self.fore_grad)
        self.grad /= input_size
        #print np.mean(self.grad)
        self.grad[:, :-1] += 2.0 * regularization_coefficient * (self.W ** 2)
        #print np.mean(2.0 * regularization_coefficient * (self.W ** 2))

    def step(self, optimizer):
        if optimizer.optimizer == "vanilla":
            self.W -= self.grad[:, :-1] * optimizer.learning_rate
            self.b -= self.grad[:, -1] * optimizer.learning_rate
        elif optimizer.optimizer == "adagrad":
            self.accumulated_gradient += self.grad ** 2
            self.W -= self.grad[:, :-1] * optimizer.learning_rate / np.sqrt(self.accumulated_gradient[:, :-1])
            self.b -= np.reshape(self.grad[:, -1], (self.output_feature_size, 1)) * optimizer.learning_rate / np.reshape(np.sqrt(self.accumulated_gradient[:, -1]), (self.output_feature_size,1))
        elif optimizer.optimizer == "adadelta":
            self.accumulated_gradient = optimizer.decay_coefficient * self.accumulated_gradient + (1.0 - optimizer.decay_coefficient) * (self.grad ** 2)
            delta = - np.sqrt(self.accumulated_updates) + optimizer.epsilon / np.sqrt(self.accumulated_gradient + optimizer.epsilon) * self.grad
            self.accumulated_updates = optimizer.decay_coefficient * self.accumulated_updates + (1.0 - optimizer.decay_coefficient) * (delta ** 2)
            self.W += delta[:, :-1]
            self.b += np.reshape(delta[:, -1], (np.shape(self.b)[0], 1))
        elif optimizer.optimizer == "adam":
            self.biased_first_moment_estimate = optimizer.exponential_decay_rate1 * self.biased_first_moment_estimate + (1.0 - optimizer.exponential_decay_rate1) * self.grad
            self.biased_second_raw_moment_estimate = optimizer.exponential_decay_rate2 * self.biased_second_raw_moment_estimate + (1.0 - optimizer.exponential_decay_rate2) * (self.grad ** 2)
            optimizer.first_moment_power *= optimizer.exponential_decay_rate1
            optimizer.second_moment_power *= optimizer.exponential_decay_rate2
            self.bias_corrected_first_moment_estimate = self.biased_first_moment_estimate / (1.0 - optimizer.first_moment_power)
            self.bias_corrected_second_moment_estimate = self.biased_second_raw_moment_estimate / (1.0 - optimizer.second_moment_power)
            self.W -= optimizer.step_size * self.bias_corrected_first_moment_estimate[:, :-1] / np.sqrt(self.bias_corrected_second_moment_estimate[:, :-1] + optimizer.epsilon)
            self.b -= optimizer.step_size * np.reshape(np.transpose(self.bias_corrected_first_moment_estimate[:, -1]) / np.sqrt(np.transpose(self.bias_corrected_second_moment_estimate[:, -1]) + optimizer.epsilon), (self.output_feature_size, 1))
        

