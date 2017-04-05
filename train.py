import numpy as np
import pandas as pd
import sys
import argparse
import json
import os
from time import sleep
from datetime import datetime
from time import sleep
import subprocess
from utils import *
from optimizer import *
from nn_model import *

parser = argparse.ArgumentParser(description='Process the data and parameters.')
# Data 
parser.add_argument('--train_data', default="train_data.csv", help='the path of train data.')
parser.add_argument('--optimizer_file', default="./optimizers.json", help='the path of the optimizer file.(JSON)')
parser.add_argument('--layer_file', default="./layers.json", help='the path of the layer file.(JSON)')
# Optimizer
parser.add_argument('--num_epoch', default=2000, type=int, help="the number of epochs will be run.")
parser.add_argument('--normalization', default=True, type=bool, help="normalization.")
parser.add_argument('--normalize_data_path', default="normalize_data.txt", help="the path of normalize data (if done).")
# Model Storing
parser.add_argument('--model_dir', default="model", help="the directory models stored.")
parser.add_argument('--log_path', default="info", help="the path of the log file.")
# Others)
parser.add_argument('--eval_step', default=20, help="the eval steps.")
parser.add_argument('--full_train', default=False, help="whether train with the whole train data or not.")
parser.add_argument('--valid_round', default=10, help="the count of valid rounds.")
args = parser.parse_args()

# Load data. 
data = np.loadtxt(args.train_data, delimiter=',')
#test_data = np.loadtxt(args.test_data, delimiter=',')
train_size = np.shape(data)[0]
feature_size = np.shape(data)[1] - 1
start_time = datetime.now()
timestamp = str(start_time.year) + str(start_time.month) + str(start_time.day) + str(start_time.hour) + str(start_time.minute) + str(start_time.second)
if not os.path.exists(args.model_dir): os.mkdir(args.model_dir)
model_path = args.model_dir + "/{}_model.json".format(timestamp)
log_path = args.log_path
num_epoch = args.num_epoch
eval_step = args.eval_step
valid_round = args.valid_round

if args.normalization == True:
    normalization(
          train_data=data, 
            #test_data=test_data,
            train_size=train_size,
            feature_size=feature_size,
            args=args)

optimizers = initialize_optimizers(args.optimizer_file)
layers = initialize_layers(args.layer_file, optimizers)

if args.full_train == False:
    full_valid_losses = np.zeros((int(num_epoch / eval_step)), dtype=np.float64)
    np.random.seed(7122)
    for i in range(valid_round):
        model = NN_model(layers, optimizers)
        valid_losses = np.zeros((int(num_epoch / eval_step)), dtype=np.float64)
        np.random.shuffle(data)
        train_data = data[:int(train_size * 0.8)]
        valid_data = data[int(train_size * 0.8):]
        for t in range(num_epoch):
            model.step(np.transpose(train_data[:, :-1]), np.transpose(train_data[:, -1]))
            #exit()
            if (t+1) % eval_step == 0: 
                valid_losses[int((t+1)/eval_step)-1] = model.eval(np.transpose(valid_data[:, :-1]), np.transpose(valid_data[:, -1]))
                print_info("Epoch {} Validation: {}".format(t+1, valid_losses[int((t+1)/eval_step)-1]))
        full_valid_losses += valid_losses
    full_valid_losses /= valid_round
    for i in range(int(num_epoch / eval_step)):
        info_string = "{}, {}, {}, {}\n".format(timestamp, feature_size, (i+1)*eval_step, full_valid_losses[i])
        with open(log_path, "a") as file: file.write(info_string)

else:
    model = NN_model(layers, optimizers)
    for t in range(num_epoch): 
        print "Epoch {}".format(t+1),
        model.step(np.transpose(data[:, :-1]), np.transpose(data[:, -1]))
    Model = dict()
    Model['layers'] = []
    for layer in model.layers:
        Layer = dict()
        Layer['W'] = layer.W.tolist()
        Layer['b'] = layer.b.tolist()
        Layer['activation_function'] = layer.activation_function
        Model['layers'].append(Layer)
    with open(model_path, "w") as file: json.dump(Model, file, indent=2)

#os.rename(args.optimizer_file, "{}/{}_optimizers.json".format(args.model_dir, timestamp))
