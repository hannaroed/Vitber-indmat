from tqdm import trange
from neural_network import NeuralNetwork
from layers import EmbedPosition, TransformerBlock, LinearLayer, CrossEntropy, Adam, Softmax
from data_generators import get_train_test_sorting

def make_model(r=5, d=10, m=2, L=2, p=15, k=5) -> NeuralNetwork:
    n_max = 2 * r - 1

    model_layers = []

    model_layers.append(EmbedPosition(n_max=n_max, m=m, d=d))
    for _ in range(L):
        model_layers.append(TransformerBlock(d=d, k=k, p=p))
    
    model_layers.append(LinearLayer(d, m))  # Unembedding
    model_layers.append(Softmax())

    model = NeuralNetwork(model_layers)

    return model


def training(model, loss_function, optimizer, data_set, n_epochs=300):
    """Training of neural network in batches"""

    # cross_entropy = layers.CrossEntropy(layers.Layer)

    for epoch in trange(n_epochs):
        for batch_idx, (x_true, y_true) in enumerate(data_set):
            Y_pred = model.forward(x_true)
            loss = loss_function.forward(Y_pred, y_true)
            dL_dY = loss_function.backward()
            model.backward(dL_dY)
            for layer in model.layers:
                if hasattr(layer, 'step_gd'):
                    layer.step_gd(optimizer)
    return model

