# NeuralNetwork
a python implement of vanilla neural network

Prerequisites
-------------
Python 2.7+  
NumPy

Usage
-----
Expected files: train_data.csv, test_data.csv, layers.json, optimizers.json  
The expected format of train_data.csv is (without the header and row index):  
```
        feature_1 feature_2 feature_3 ... value/label
data_1     ...      ...       ...             ...
data_2     ...      ...       ...             ...
data_3     ...      ...       ...             ...
.
.
.
```
and the expected format of test_data.csv is (without the header and row index):
```
        feature_1 feature_2 feature_3 
data_1     ...      ...       ...    
data_2     ...      ...       ...      
data_3     ...      ...       ...      
.
.
.
```
The default loss function is cross entropy loss, other functions will be added in future. Please feel free to modify the code of the function ``cal_loss_grad``(in nn_layer.py).    
The expected format of layers.json is:
```
{
  "layers": [
    {
      "activation_function"="..."
      "input_feature_size"= N
      "output_feature_size"= M
    },
    ...
    ...
    ...
    }
  ]
}
```
and the expected format of optimizers.json is:
```
{
  "optimizers": [
    {
      "optimizer":"..."
      ...
      ...
      ...
      "regularization_coefficient="..."
    },
    ...
    ...
    ...
    }
  ]
}
```
The default parameters of the optimizers will be added in future, and for now you have to declare all the parameters that the optimizers need in the ``optimizers.json``. To get the information about the optimizers and parameters you can modify, please take a look at ``optimizers.py``.    
To run the training code:
```
$ python train.py [--train_data train_data.csv] \  
[--optimizer_file optimizers.json] [--layers_file layers.json] \  
[--num_epoch 2000] [--eval_step 20] [--full_train False] [--valid_round 20] \  
[--normalization True] [--normalize_data_path normalize_data.txt] \  
[--model_dir model/] [--log_path info]
```
If you set the ``full_train`` argument to False, it wouldn't produce the model file, and it would train with 80% training data, with the other 20% data being used for validation, and it will run ``valid_round`` times, storing the result in the ``log_path``.  
To run the testing code:
```
$ python test.py [--test_data test_data.csv] [--normalize_data normalize_data.txt] \  
[--answer_path ./answer.csv] --model (the model file)
```
The output of the testing code will be like:
```
id,label
0,...
1,...
...
...
...
```
Please feel free to modify the code of ``test.py``.
