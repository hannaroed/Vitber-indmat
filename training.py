from tqdm import trange
from neural_network import NeuralNetwork
from optimizer import Adam
from layers import EmbedPosition, TransformerBlock, LinearLayer, CrossEntropy, Softmax
from data_generators import get_train_test_sorting, get_train_test_addition
from utils import onehot
import numpy as np


def make_model(r=5, d=10, m=2, L=5, p=128, k=128) -> NeuralNetwork:
    """

    r: int, number of elements in the input sequence
    d: int, embedding size
    m: int, number of elements in the vocabulary
    k: int, dimension size for query, key and value vectors
    p: int, hidden dimension for feed forward network
    """
    # n_max = 2 * r - 1

    # model_layers = []

    # model_layers.append(EmbedPosition(n_max, m, d, 0.1))
    # for _ in range(L):
    #     model_layers.append(TransformerBlock(d, k, p))
    
    # model_layers.append(LinearLayer(d, m))  # Unembedding
    # model_layers.append(Softmax())

    model = NeuralNetwork(r, d, m, L, p, k)

    return model


def training_sorting(model: NeuralNetwork, loss_function: CrossEntropy, optimizer: Adam, data_set, m, n_epochs=300):
    """Training of neural network in batches"""

    x_train, y_train = data_set['x_train'], data_set['y_train']
    mean_loss_arr = np.zeros(n_epochs)

    for epoch in trange(n_epochs):
        for batch_idx in range(x_train.shape[0]):
            x = x_train[batch_idx]
            y_true = y_train[batch_idx]

            x = onehot(x, m)
            Y_pred = model.forward(x)
            loss = loss_function.forward(Y_pred, y_true)
            dL_dY = loss_function.backward()
            model.backward(dL_dY)
            model.step_gd(optimizer)

        mean_loss = np.mean(loss)
        mean_loss_arr[epoch] = mean_loss
        #print("Iteration ", str(epoch), " L = ", mean_loss, "")

    return model, mean_loss_arr

def training_addition(model, loss_function, optimizer, data_set, m, n_epochs=300, r=2):
    """Training of neural network in batches"""

    x_train, y_train = data_set['x_train'], data_set['y_train']
    mean_loss_arr = np.zeros(n_epochs)

    for epoch in trange(n_epochs):
        for batch_idx in range(x_train.shape[0]):
            x = x_train[batch_idx]
            y_true = y_train[batch_idx]

            x = onehot(x, m)
            Y_pred = model.forward(x)
            first_part = Y_pred[:,:,:-(r+1)]
            last_part_rev = Y_pred[:,:,-(r+1):][:, :, ::-1]
            #print(first_part.shape, last_part_rev.shape)
            Y_pred_new = np.concatenate((first_part, last_part_rev), axis=2)
            loss = loss_function.forward(Y_pred_new, y_true)
            dL_dY = loss_function.backward()
            model.backward(dL_dY)
            model.step_gd(optimizer)

        mean_loss = np.mean(loss)
        mean_loss_arr[epoch] = mean_loss
        #print("Iteration ", str(epoch), " L = ", mean_loss, "")

    return model, mean_loss_arr