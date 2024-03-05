import numpy as np


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

    print(train_set[0][0].shape)

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
    print(train_set[1][0])
    loss_function.forward(y_hat, y_true=train_set[1][0])
    grad_loss = loss_function.backward()

    model.backward(grad_loss)
    

def test_adam():
    np.seterr(all='raise')
    model = make_model()
    optimizer = l.Adam()

    grad_loss = np.random.randn(250)

    train_set = get_test_data()
    batch_x = onehot(train_set[0][0], 2)
    y_hat = model.forward(batch_x)

    loss_function = l.CrossEntropy()
    print(train_set[1][0])
    loss_value = loss_function.forward(y_hat, y_true=train_set[1][0])
    print(f'{loss_value=}')
    print(f'{loss_value.shape=}')
    grad_loss = loss_function.backward()

    model.backward(grad_loss)
    for layer in model.layers:
        if hasattr(layer, 'step_gd'):
            layer.step_gd(optimizer)


if __name__ == '__main__':
    # test_forward_shape()
    # test_backward()
    test_adam()