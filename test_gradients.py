import numpy as np
from tqdm import trange
from typing import Any

import layers as l
from utils import onehot
from training import make_model, training
from data_generators import get_train_test_sorting


def get_test_data():
    data_set = get_train_test_sorting(length=5, num_ints=2, samples_per_batch=250, n_batches_train=10, n_batches_test=2)
    train_set = list(zip(data_set['x_train'], data_set['y_train']))
    return train_set

def test_forward_shape():
    model = make_model()

    train_set = get_test_data()


    batch_x = onehot(train_set[0][0], 2)
    out = model.forward(batch_x)
    assert out.shape == (250, 2, 9)


def test_backward():
    model = make_model()

    grad_loss = np.random.randn(250)

    train_set = get_test_data()
    batch_x = onehot(train_set[0][0], 2)
    y_hat = model.forward(batch_x)

    loss_function = l.CrossEntropy()
    loss_function.forward(y_hat, y_true=train_set[1][0])
    grad_loss = loss_function.backward()

    model.backward(grad_loss)
    

def test_adam():
    np.seterr(all='raise')

    # Initialize model and optimizer
    model = make_model()
    optimizer = l.Adam()
    # Overfit on a single example

    # Get all training data
    train_set = get_test_data()

    loss_function = l.CrossEntropy()

    m = 2

    # First input value from training set
    input = train_set[0][0]
    output = train_set[0][1]
    batch_x = onehot(input, m)

    for _ in range(1000):
        y_hat = model.forward(batch_x)
        # y_hat: (b, m, n)
        y_hat_indices = np.argmax(y_hat, axis=1)

        correct = y_hat_indices == output
        accuracy = np.mean(correct)

        # y_true is not one-hot encoded, but instead class indices
        loss_value = loss_function.forward(y_hat, y_true=train_set[1][0]).mean()

        # dLdY: (b, m, n)
        grad_loss = loss_function.backward()


        model.backward(grad_loss)

        model.step_gd(optimizer)

        print(f'{accuracy=:.5f}, {loss_value=:.5f}')


def module_backward_works(input, out_shape: tuple, module):
    # Not done, work in progress

    # Want dY/dX of this value
    grad_output = np.ones(out_shape)

    # Compute the forward pass
    forward_result = module.forward(input)

    # Now do backward with this in mind
    dL_dx = module.backward(grad_output)

    perturbed = input + delta_input
    forward_perturbed = module.forward(perturbed)
    # print(forward_perturbed)
    print(((forward_perturbed - forward_result).sum() / delta))
    # print(grad_output)
    # assert np.allclose((forward_perturbed - forward_result) / delta, grad_output, atol=1e-6)
    

def test_backward_correct():
    batch_size = 10
    in_dims = 2
    out_dims = 3
    seq_len = 5
    module = l.LinearLayer(in_dims, out_dims, has_bias=False)
    input = np.random.randn(batch_size, in_dims, seq_len)
    module_backward_works(input, (batch_size, out_dims, seq_len), module)


if __name__ == '__main__':
    # test_forward_shape()
    # test_backward()
    test_adam()
    # test_backward_correct()