import numpy as np
from neural_network import NeuralNetwork
import layers

def onehot(x,m):
    """
    Input:
    - x : np.array of integers with shape (b,n)
             b is the batch size and 
             n is the number of elements in the sequence
    - m : integer, number of elements in the vocabulary 
                such that x[i,j] <= m-1 for all i,j

    Output:     
    - x_one_hot : np.array of one-hot encoded integers with shape (b,m,n)

                    x[i,j,k] = 1 if x[i,j] = k, else 0 
                    for all i,j
    """

    b,n = x.shape

    #Making sure that x is an array of integers
    x = x.astype(int)
    x_one_hot = np.zeros((b,m,n))
    x_one_hot[np.arange(b)[:,None],x,np.arange(n)[None,:]] = 1
    return x_one_hot


def training(loss, theta, alpha, beta1, beta2, data_set):
    """Training of neural network in batches"""

    network = NeuralNetwork()
    cross_entropy = layers.CrossEntropy(layers.Layer)

    for j in range(0, 300):
        for k in range(0,len(data_set)):
            x_true, y_true = data_set[k]
            Y_pred = network.forward(x_true)
            loss[j][k] = cross_entropy.forward(Y_pred, y_true)
            for layer in network.layers:
                if isinstance(layer, layers.LinearLayer, layers.EmbedPosition, layers.FeedForward, layers.Attention):
                    grad = network.backward()
                    W_new = layers.Adam.step_adam(grad, layer, beta1, beta2, alpha)
    return W_new