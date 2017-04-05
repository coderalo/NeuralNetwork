from nn_model import *
import numpy as np

parser = argparse.ArgumentParser(description='Process the data and parameters.')
parser.add_argument('--test_data', default="parse_test_data.csv", help='the path of test data.')
parser.add_argument('--normalize_data', default="normalize_data", help='the path of normalize data.')
parser.add_argument('--answer_path', default="./answer.csv", help='the path of answer.')
parser.add_argument('--model', help="the path of the model.")
args = parser.parse_args()

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

test_data = np.loadtxt(args.test_data, delimiter=',')
with open(args.model, "r") as file: model = json.load(file)
normalize_data = np.loadtxt(args.normalize_data, delimiter=',')
mean = np.array(normalize_data[0])
std = np.array(normalize_data[1])

test_data = np.transpose((test_data - mean) / std)
print np.shape(test_data)
for layer in model['layers']: 
    print(np.shape(np.array(layer['W'])), np.shape(np.array(layer['b'])))
    test_data = np.matmul(np.array(layer['W']), test_data) + np.array(layer['b'])
    if layer['activation_function'] == "unit": test_data = np.copy(test_data)
    elif layer['activation_function'] == "tanh": test_data = np.tanh(test_data)
    elif layer['activation_function'] == "sigmoid": test_data = sigmoid(test_data)
    elif layer['activation_function'] == 'relu': test_data = np.maximum(test_data, 0)


test_data = np.round(test_data)
with open(args.answer_path, "w") as file:
    file.write("id,label\n")
    for i, val in enumerate(test_data[0]): file.write(str(i+1) + "," + str(int(val)) + "\n")
