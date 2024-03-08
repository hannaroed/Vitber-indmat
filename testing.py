from tqdm import trange
from neural_network import NeuralNetwork
from optimizer import Adam
from layers import EmbedPosition, TransformerBlock, LinearLayer, CrossEntropy, Softmax
from data_generators import get_train_test_sorting, get_train_test_addition
from utils import onehot
import numpy as np
from layers import numba_max_axis1

def test_sorting(trained_model, data_set, m):
    """Testing of neural network in batches"""
    x_test, y_test = data_set['x_test'], data_set['y_test']
    counter = 0
    
    for i in range(x_test.shape[0]):
        for batch_idx in range(x_test.shape[1]):
            x = x_test[batch_idx]
            y_true = y_test[batch_idx]
            
            X = onehot(x, m)
            Y_pred = trained_model.forward(X)
            print('Y PRED SHAPE:')
            print(Y_pred.shape)
            print(Y_pred)
            print('Y TEST SHAPE:')
            print(y_test.shape)

            if (Y_pred[batch_idx].all() == y_true[batch_idx][i].all()):
                counter+=1

    correct_percentage = (counter/x_test.shape[2])*100
  
    return correct_percentage