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

    # The entire dataset is of shape x: (num_batches, batch_size, sequence_length)
    # y: (num_batches, batch_size, sequence_length)

    # Keep track of all the accuracies
    total_correct, total = 0, 0

    for batch_idx in range(x_test.shape[0]):
        # A batch is a group of batch_size samples
        # Each sample is a sequence of elements
        # Each element is an integer in the range [0, m-1]

        x = x_test[batch_idx]  # Get one batch from the dataset, shape (batch_size, sequence_length)
        y_true = y_test[batch_idx]  # Get the corresponding labels, shape (batch_size, out_sequence_length)
        
        # Turn the input sequence into a one-hot encoded input of shape (batch_size, m, sequence_length)
        # This sequence has m "channels", one for each element in the vocabulary
        # Element [i, j, k] is 1 if the k-th element of the i-th sample sequence is j, 0 otherwise
        X = onehot(x, m)

        # The predictions are of shape (batch_size, m, sequence_length) because they contain the scores
        Y_pred = trained_model.forward(X)
        print('Y PRED SHAPE:')
        print(Y_pred.shape)
        print('Y TEST SHAPE:')
        print(y_test.shape)

        # This is wrong, let me explain:
        # batch_idx is the index number of the batch we are looking at. Only one batch gets put through the network at a time,
        # so it doesn't make sense to index the output with the batch index
        # Further, .all() is a method that returns True (just one boolean, not an array) if all elements in the array are True, and False otherwise.
        # This array is not a boolean array.
        # if (Y_pred[batch_idx].all() == y_true[batch_idx][i].all()):
        #     counter+=1

        # We need to compare the predictions to what we were expecting to see
        # Our output is a set of scores for each element in the vocabulary, for each position in the sequence
        # The answer we produced is the element index with the highest score at each position in the sequence
        # We compute this by finding the argmax (the index of the highest element) in the sequence
        # Remember that Y_pred is of shape (batch_size, _m_, sequence_length), so the index we find the argmax over is 1
        y_hat = np.argmax(Y_pred, axis=1)  # Shape (batch_size, sequence_length)
        
        # Now we have an array that contains the guesses that our model made.
        # We can compare this to the ground truth from our dataset

        is_correct_guess = y_hat == y_true  # Shape (batch_size, sequence_length)
        # The array above is a boolean array, that has a True value at each position where the guess was correct, and a False value otherwise

        # True values are treated as 1, and False values as 0, so we can sum to get the number of correct guesses
        total_correct += is_correct_guess.sum()
        total += is_correct_guess.size

    # correct_percentage = (counter/x_test.shape[2])*100

    total_percentage = total_correct / total * 100
  
    return total_percentage