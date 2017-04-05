import numpy as np
import pandas as pd
import sys
import argparse
import json
from time import sleep
from datetime import datetime
from time import sleep
import subprocess
from optimizer import *
from nn_layer import *

def print_info(info):
    time = datetime.now()
    print '[{:0>2}:{:0>2}:{:0>2}]: {}'.format(time.hour, time.minute, time.second, info)

def normalization(train_data, train_size, feature_size, args):
    mean = np.mean(train_data[:, :-1], axis=0)
    std = np.std(train_data[:, :-1], axis=0)
    train_data[:, :-1] = (train_data[:, :-1] - mean) / std
    #for i in range(test_size): test_data[i] = (test_data[i]-mean[:-1])/standard[:-1]
    normalize_data = np.zeros((2, feature_size))
    normalize_data[0] = mean
    normalize_data[1] = std
    np.savetxt(args.normalize_data_path, normalize_data, delimiter=',')

def initialize_optimizers(optimizer_file_path):
    optimizers = []
    with open(optimizer_file_path, "r") as file: optimizers_data = json.load(file)
    for optimizer in optimizers_data['optimizers']: optimizers.append(Optimizer(optimizer))
    return optimizers

def initialize_layers(layer_file_path, optimizers):
    layers = []
    with open(layer_file_path, "r") as file: layers_data = json.load(file)
    for i, layer in enumerate(layers_data['layers']): layers.append(NN_layer(layer, optimizers[i]))
    return layers

