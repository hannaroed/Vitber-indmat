from tqdm import trange
from neural_network import NeuralNetwork
from optimizer import Adam
from layers import CrossEntropy, jit_onehot
import numpy as np


def make_model( r: int = 5, d: int = 10, m: int = 2, L: int = 5, p: int = 128, k: int = 128) -> NeuralNetwork:
    '''
    r: int, number of elements in the input sequence
    d: int, embedding size
    m: int, number of elements in the vocabulary
    k: int, dimension size for query, key and value vectors
    p: int, hidden dimension for feed forward network
    '''

    # Creates a NeuralNetwork with all the layers.
    model = NeuralNetwork(r, d, m, L, p, k)

    return model


def training_sorting(model: NeuralNetwork, loss_function: CrossEntropy, optimizer: Adam, data_set: dict, m: int, n_epochs: int = 300):
    '''
    Training on sorting integers in batches for a neural network.

    '''

    # Getting training data
    x_train, y_train = data_set['x_train'], data_set['y_train']

    mean_loss_arr = np.zeros(n_epochs)

    # Making progress bar
    pbar = trange(n_epochs, desc='Training model')

    for epoch in pbar:
        correct = 0
        total = 0
        for batch_idx in range(x_train.shape[0]):
            x = x_train[batch_idx]
            y_true = y_train[batch_idx]

            # Taking one hot of y_true and padding it so we can compare it with y_pred in our loss function
            Y_true = jit_onehot(y_true, m)
            Y_true_pad = np.pad(Y_true, ((0, 0), (0, 0), (x.shape[1] - Y_true.shape[2], 0)))

            X = jit_onehot(x, m)
            Y_pred = model.forward(X)
            Y_pred_slice = Y_pred[:, :, -Y_true.shape[2]:]

            # Summing up correct guesses by comparing y_true with a sliced Y_pred     
            correct += np.sum(np.argmax(Y_pred_slice, axis=1) == y_true)

            total += y_true.size

            # Computing the loss
            loss = loss_function.forward(Y_pred, Y_true_pad)

            dL_dY = loss_function.backward()
            model.backward(dL_dY)
            model.step_gd(optimizer)

        # Finding the mean loss
        mean_loss = np.mean(loss)
        mean_loss_arr[epoch] = mean_loss

        # Progress bar with current mean of loss and accuracy
        pbar.set_postfix({'loss': mean_loss, 'accuracy': correct / total})

    return model, mean_loss_arr


def training_addition(model: NeuralNetwork, loss_function: CrossEntropy, optimizer: Adam, data_set: dict, m: int, n_epochs: int = 300):
    '''
    Training on addition in batches for a neural network.

    '''
    
    # Getting training data
    x_train, y_train = data_set['x_train'], data_set['y_train']

    mean_loss_arr = np.zeros(n_epochs)

    # Making progress bar
    pbar = trange(n_epochs, desc='Training model')

    for epoch in pbar:
        correct = 0
        total = 0
        for batch_idx in range(x_train.shape[0]):   
            x = x_train[batch_idx]
            y_true = y_train[batch_idx]
                      
            Y_true = jit_onehot(y_true, m)
            X = jit_onehot(x, m)
            Y_pred = model.forward(X)
            
            # Slicing our prediction to count correct guesses
            Y_pred_slice = Y_pred[:,:,-Y_true.shape[2]:]

            # Padding Y_true so we can compare it with Y_pred in our loss function
            Y_true_pad = np.pad(Y_true, ((0, 0), (0, 0), (Y_pred.shape[2] - Y_true.shape[2], 0)))
            
            # Comparing y_true with a sliced Y_pred     
            correct += np.sum(np.argmax(Y_pred_slice, axis=1) == y_true)
            total += y_true.size

            # Computing the loss
            loss = loss_function.forward(Y_pred, Y_true_pad)

            dL_dY = loss_function.backward()
            model.backward(dL_dY)
            model.step_gd(optimizer)

        # Finding the mean loss
        mean_loss = np.mean(loss)
        mean_loss_arr[epoch] = mean_loss

        # Progress bar with current mean of loss and accuracy
        pbar.set_postfix({'loss': mean_loss, 'accuracy': correct / total})

    return model, mean_loss_arr