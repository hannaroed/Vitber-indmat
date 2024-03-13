from neural_network import NeuralNetwork
from utils import jit_onehot
import numpy as np
from tqdm.auto import trange


def test_sorting(trained_model: NeuralNetwork, data_set, m, r):
    '''
    Testing sorting of neural network in batches.
    
    '''
    x_test, y_test = data_set['x_test'], data_set['y_test']

    # The entire dataset is of shape:
    # x: (num_batches, batch_size, sequence_length)
    # y: (num_batches, batch_size, sequence_length)

    # Keeping track of all the accuracies
    total_correct, total = 0, 0

    for batch_idx in trange(x_test.shape[0]):
        # A batch is a group of batch_size samples
        # Each sample is a sequence of elements
        # Each element is an integer in the range [0, m-1]

        # Get one batch from the dataset, shape (batch_size, sequence_length)
        x = x_test[batch_idx]  
        y_true = y_test[batch_idx]

        # Keeping track of current x
        x_current = x

        for _ in range(y_true.shape[1]):
            prediction = trained_model.forward(jit_onehot(x_current, m))
            prediction_value = np.argmax(prediction, axis=1)[:, -1:]
            x_current = np.concatenate((x_current, prediction_value), axis=1)
        
        y_hat = x_current[:, -y_true.shape[1]:]

        # Checks if guess is correct
        is_correct_guess = y_hat == y_true # Shape (batch_size, sequence_length)

        # Sum up to get the number of correct guesses
        total_correct += is_correct_guess.sum()
        total += is_correct_guess.size

    # Calculate percentage
    total_percentage = total_correct / total * 100
  
    return total_percentage

def test_addition(trained_model: NeuralNetwork, data_set, m):
    '''
    Testing addition of neural network in batches.
    
    '''

    x_test, y_test = data_set['x_test'], data_set['y_test']

    # Keeping track of all the accuracies
    total_correct, total = 0, 0

    for batch_idx in range(x_test.shape[0]):
        x = x_test[batch_idx]  # Get one batch from the dataset, shape (batch_size, sequence_length)
        y_true = y_test[batch_idx]  # Shape (batch_size, out_sequence_length)
        
        # Keeping track of current x
        x_current = x
        for _ in range(y_true.shape[1]):
            prediction = trained_model.forward(jit_onehot(x_current, m))
            prediction_value = np.argmax(prediction, axis=1)[:, -1:]
            x_current = np.concatenate((x_current, prediction_value), axis=1)
        
        # The predictions are of shape (batch_size, m, sequence_length)
        y_hat = x_current[:, -y_true.shape[1]:]
    
        # Checks if guess is correct
        is_correct_guess = np.flip(y_hat, axis=1) == y_true  # Shape (batch_size, sequence_length)

        # Sum up to get the number of correct guesses
        total_correct += is_correct_guess.sum()
        total += is_correct_guess.size

    # Calculate percentage
    total_percentage = total_correct / total * 100
  
    return total_percentage