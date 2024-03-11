from __future__ import annotations
import numpy as np
from utils import onehot
from math import sqrt
from numba import njit
import numba as nb
from numba.experimental import jitclass
from optimizer import Optimizer

@njit(inline='always')
def numba_max_axis1(a):
    # Keeps dims
    assert a.ndim == 3
    running = np.zeros((a.shape[0], 1, a.shape[2]))
    for i in range(a.shape[1]):
        running = np.maximum(running, a[:, i:i+1, ...])
    return running

@njit(inline='always')
def numba_mean_axis0(a):
    # Reduces dims
    assert a.ndim == 3
    running = np.zeros((a.shape[1], a.shape[2]))
    for i in range(a.shape[0]):
        running += a[i, ...]
    return running / a.shape[0]

@njit(inline='always')
def numba_mean_bias(grad_bias):
    out = np.zeros(grad_bias.shape[1])
    for i in range(grad_bias.shape[0]):
        for j in range(grad_bias.shape[1]):
            for k in range(grad_bias.shape[2]):
                out[j] += grad_bias[i, j, k]
    return out
    

class Layer:
    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    
    def step_gd(self, optimizer: Optimizer): 
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                'm': m,         The first moment of the gradient
                'v': v          The second moment of the gradient
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        if not hasattr(self, 'params'):
            return

        for param in self.params.values():
            optimizer.update(param)


key_type, entry_type = nb.types.unicode_type, nb.float64[:, :]
inner_type = nb.types.DictType(key_type, entry_type)
outer_type = nb.types.DictType(key_type, inner_type)

# jitclass compiles all the methods in the class to make it much faster
@jitclass([
    ('params', outer_type),
    ('x', nb.float64[:, :, :]),
    ('has_bias', nb.boolean),
])
class LinearLayer(Layer):
    """
    Linear Layer
    """
    def __init__(self, input_size, output_size, has_bias, init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale

        w = np.random.randn(output_size,input_size)*init_scale
        inner = nb.typed.Dict.empty(key_type, entry_type)
        inner['w'] = w
        inner['d'] = np.zeros_like(w)
        params = nb.typed.Dict.empty(key_type, inner_type)
        self.params = params
        self.params['w'] = inner
        self.has_bias = has_bias

        if has_bias:
            inner = nb.typed.Dict.empty(key_type, entry_type)
            bound = 1 / sqrt(input_size)
            inner['w'] = np.random.uniform(-bound, bound, (output_size, 1))
            inner['d'] = np.zeros_like(inner['w'])
            self.params['b'] = inner

        # self.params = {"w":{'w':w,
        #                     'd':np.zeros_like(self.w), }}
        
    def forward(self, x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        self.x = x
        
        #Return output of layer
        # Original
        # y = np.einsum('od,bdn->bon', self.params['w']['w'], x)

        # Efficient
        # w = self.params['w']['w']
        # w = w[None, :, :]
        # y = w @ x

        # Numba
        w = self.params['w']['w']
        y = batched_mm(w, x)
        if self.has_bias:
            b = self.params['b']['w']
            y += b[None, :, :]
        return y
        
    def backward(self, grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        #Compute gradient (average over B batches) of loss wrt weight w: 

        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)

        # Original
        # grad: (b, o, n)
        # self.x: (b, d, n)
        # out: (o, d)
        # self.params['w']['d'] = np.einsum('bon,bdn->od', grad, self.x) / b

        # Numba
        grad_mean = numba_mean_axis0(grad)
        x_mean = numba_mean_axis0(self.x).T.copy()
        self.params['w']['d'] = grad_mean @ x_mean

        if self.has_bias:
            self.params['b']['d'] = numba_mean_bias(grad)[:, None]

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        # Original
        # return np.einsum('od,bon->bdn', self.params['w']['w'], grad)

        # Optimized
        # out = self.params['w']['w'].T @ grad

        # Numba
        w = self.params['w']['w']
        out = batched_mm(w.T.copy(), grad)
        return out
    

@njit
def make_D_matrix(n):
    D = np.zeros((n, n))
    i1, i2 = np.tril_indices(n, -1)
    for i, j in zip(i1, i2):
        D[i, j] = -np.inf
    return D[None, :, :]

# @njit(inline='always')
# def double_batched_mm(A, B):
#     ab, ao, ai = A.shape
#     bb, bi, bo = B.shape
#     assert ai == bi
#     assert ab == bb
#     out = np.zeros((ab, ao, bo))
#     for i in range(ab):
#         out[i] = A[i] @ B[i]
#     return out

@njit(parallel=True)
def batched_mm(A, B):
    """Either A or B or both can have a batch dimension"""
    if A.ndim == 3 and B.ndim == 3:
        ab, ao, ai = A.shape
        bb, bi, bo = B.shape
        assert ab == bb
        assert ai == bi
        out = np.zeros((ab, ao, bo))
        for i in nb.prange(ab):
            out[i] = A[i] @ B[i]
        return out
    if A.ndim == 3 and B.ndim == 2:
        ab, ao, ai = A.shape
        bi, bo = B.shape
        assert ai == bi
        out = np.zeros((ab, ao, bo))
        for i in nb.prange(ab):
            out[i] = A[i] @ B
        return out
    if A.ndim == 2 and B.ndim == 3:
        ao, ai = A.shape
        ab, bi, bo = B.shape
        assert ai == bi
        out = np.zeros((ab, ao, bo))
        for i in nb.prange(ab):
            out[i] = A @ B[i]
        return out
    if A.ndim == 2 and B.ndim == 2:
        return A @ B
    raise ValueError("Invalid dimensions")

@jitclass([
    ('prev_A', nb.optional(nb.float64[:, :, :])),
    ('prev_B', nb.optional(nb.float64[:, :, :])),
])
class Matmul(Layer):
    def __init__(self):
        self.prev_A: np.ndarray | None = None
        self.prev_B: np.ndarray | None = None
    
    def forward(self, A, B):
        self.prev_A = A
        self.prev_B = B
        # Standard
        # return A @ B

        # Numba
        return batched_mm(A, B)
    
    def backward(self, dL_dAB):
        # Standard
        # dL_dA = dL_dAB @ self.prev_B.transpose(0, 2, 1)
        # dL_dB = self.prev_A.transpose(0, 2, 1) @ dL_dAB

        B_T = self.prev_B.transpose(0, 2, 1).copy()
        A_T = self.prev_A.transpose(0, 2, 1).copy()

        # Numba
        dL_dA = batched_mm(dL_dAB, B_T)
        dL_dB = batched_mm(A_T, dL_dAB)
        return dL_dA, dL_dB


@jitclass([
    ('epsilon', nb.float64),
    ('prev_Q', nb.optional(nb.float64[:, :, :])),
    ('prev_P', nb.optional(nb.float64[:, :, :])),
    ('prev_z_l', nb.optional(nb.float64[:, :, :])),
])
class Softmax(Layer):

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

        self.prev_Q: np.ndarray | None = None
        self.prev_P: np.ndarray | None = None
        self.prev_z_l: np.ndarray | None = None
    
    def forward(self, x):
        """Columnwise softmax operation"""
        # x: (batch, d, n)
        # self.x = x

        # shifted = np.where(np.isneginf(x), x, x - np.max(x, axis=1, keepdims=True))

        # Numba
        max_vals = numba_max_axis1(x)
        shifted = np.where(np.isneginf(x), x, x - max_vals)

        P = np.exp(shifted)
        Q = np.sum(P, axis=1)[:, None, ...]

        z_l = P / (Q + self.epsilon)

        self.prev_P = P
        self.prev_Q = Q
        self.prev_z_l = z_l

        return z_l

    def backward(self, grad):
        P, Q, z_l = self.prev_P, self.prev_Q, self.prev_z_l
        
        S = P / (Q * Q + self.epsilon)

        dL_dz = grad * z_l - np.sum(grad * S, axis=1)[:, None, ...] * P

        
        return dL_dz


@jitclass([
    ('softmax', Softmax.class_type.instance_type),
    ('matmul1', Matmul.class_type.instance_type),
    ('matmul2', Matmul.class_type.instance_type)
])
class Attention(Layer):
    def __init__(self):
        self.softmax = Softmax(1e-8)
        self.matmul1 = Matmul()
        self.matmul2 = Matmul()

    def forward(self, Q, K, V):
        # queries, keys, values
        # søkeverdi, sammenligningsverdi, verdi
        # For every query, compare to all keys, and take from the corresponding value
        b, d, n = Q.shape
        D = make_D_matrix(n)

        # Q = Q.transpose(0, 2, 1)
        # self.matmuli.noop()
        qk_prod = self.matmul1.forward(Q.transpose(0, 2, 1).copy(), K) / sqrt(d)

        A = self.softmax.forward(qk_prod + D)

        VA = self.matmul2.forward(V, A)

        return VA
    
    def backward(self, dL_dVA):
        dL_dV, dL_dA = self.matmul2.backward(dL_dVA)

        dL_dqk_prod = self.softmax.backward(dL_dA)

        dL_dQ, dL_dK = self.matmul1.backward(dL_dqk_prod)
        dL_dQ = dL_dQ.transpose(0, 2, 1).copy()

        b, d, n = dL_dQ.shape

        dL_dQ /= sqrt(d)
        dL_dK /= sqrt(d)

        return dL_dQ, dL_dK, dL_dV


@jitclass([
    ('W_q', LinearLayer.class_type.instance_type),
    ('W_k', LinearLayer.class_type.instance_type),
    ('W_v', LinearLayer.class_type.instance_type),
    ('W_o', LinearLayer.class_type.instance_type),
    ('prev_A', nb.optional(nb.float64[:, :, :])),
    ('softmax', Softmax.class_type.instance_type),
    ('attention', Attention.class_type.instance_type)
])
class SelfAttention(Layer):

    def __init__(self, d, k):
        self.W_q = LinearLayer(d, k, True, 0.1)
        self.W_k = LinearLayer(d, k, True, 0.1)
        self.W_v = LinearLayer(d, k, True, 0.1)
        self.W_o = LinearLayer(k, d, True, 0.1)

        self.prev_A: np.ndarray | None = None

        self.softmax = Softmax(1e-8)

        self.attention = Attention()

    def forward(self, z):
        b, k, n = z.shape

        Q = self.W_q.forward(z)  # (b, k, n)
        K = self.W_k.forward(z)  # (b, k, n)
        V = self.W_v.forward(z)  # (b, k, n)

        VA = self.attention.forward(Q, K, V)  # (b, k, n)

        out = self.W_o.forward(VA)  # (b, d, n)

        return out

    def backward(self, grad):
        dL_dVA = self.W_o.backward(grad)  # (b, k, n)

        dL_dQ, dL_dK, dL_dV = self.attention.backward(dL_dVA)  # each (b, k, n)

        dL_dz = self.W_q.backward(dL_dQ) + self.W_k.backward(dL_dK) + self.W_v.backward(dL_dV)

        return dL_dz
    
    def step_gd(self, optimizer):
        self.W_q.step_gd(optimizer)
        self.W_k.step_gd(optimizer)
        self.W_v.step_gd(optimizer)
        self.W_o.step_gd(optimizer)
    
@njit(inline='always')
def numba_mean_axis1(a):
    # Reduces dims
    running = np.zeros((a.shape[0],))
    for i in range(a.shape[1]):
        running += a[:, i]
    return running / a.shape[1]

@njit
def jit_onehot(x, m):
    b, n = x.shape
    x_one_hot = np.zeros((b, m, n))
    for i in range(b):
        for j in range(n):
            x_one_hot[i, x[i, j], j] = 1
    return x_one_hot

@jitclass([
    ('epsilon', nb.float64),
    ('prev_y_pred', nb.optional(nb.float64[:, :, :])),
    ('prev_y', nb.optional(nb.int64[:, :]))
])
class CrossEntropy(Layer):

    def __init__(self, epsilon = 10e-18):
        self.epsilon = epsilon
        self.prev_y_pred: np.ndarray | None = None
        self.prev_y: np.ndarray | None = None
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        # y_pred: (batch, m, n)
        # y_true: (batch, n)
        # m = number of classes

        # IMPORTANT -- this takes in vectors of indices, and NOT one-hot encoded vectors.
        # This is a more efficient way of doing it

        self.prev_y_pred = y_pred
        self.prev_y = y_true

        # out: (batch,)

        # batch_index = np.arange(y_pred.shape[0])[:, None]
        # seq_index = np.arange(y_pred.shape[2])[None, :]

        # per_token_loss: (batch, n)
        # Dette er raskere med numba enn med numpy
        per_token_loss = np.zeros((y_pred.shape[0], y_pred.shape[2]))
        for batch_index in range(y_pred.shape[0]):
            for seq_index in range(y_pred.shape[2]):
                per_token_loss[batch_index, seq_index] = -np.log(y_pred[batch_index, y_true[batch_index, seq_index], seq_index] + self.epsilon)
        # per_token_loss = -np.log(y_pred[batch_index, y_true, seq_index] + self.epsilon)
        
        # per_sequence_loss: (batch,)

        per_sequence_loss = numba_mean_axis1(per_token_loss)
        # per_sequence_loss = per_token_loss.mean(axis=1)

        return per_sequence_loss

    def backward(self):
        b, m, n = self.prev_y_pred.shape

        # Create one-hot
        prev_y_oh = jit_onehot(self.prev_y, m)

        out = - 1/n * (prev_y_oh / (self.prev_y_pred + self.epsilon))

        return out


@jitclass([
    ('x', nb.float64[:, :, :]),
])
class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        pass

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))


@jitclass([
    ('embed', LinearLayer.class_type.instance_type),
    ('w', nb.float64[:, :]),
    ('params', outer_type)
])
class EmbedPosition(Layer):
    def __init__(self, n_max, m, d, init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,False,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max) * init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = nb.typed.Dict.empty(key_type, inner_type)
        self.params['Wp'] = nb.typed.Dict.empty(key_type, entry_type)
        self.params['Wp']['w'] = self.w
        self.params['Wp']['d'] = np.zeros_like(self.w)
        # self.params = {"Wp": {'w':self.w, 'd':None}}

    def forward(self, X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self, grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(self.w)
        self.params['Wp']['d'] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self, optimizer):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(optimizer)

        # Update parameters
        for param in self.params.values():
            optimizer.update(param)


@jitclass([
    ('l1', LinearLayer.class_type.instance_type),
    ('activation', Relu.class_type.instance_type),
    ('l2', LinearLayer.class_type.instance_type),
    ('x', nb.optional(nb.float64[:, :, :])),
])
class FeedForward(Layer):
    def __init__(self, d, p, init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        self.x: np.ndarray | None = None

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,True,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,True,init_scale)

    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        out = x + self.l2.forward(self.activation.forward(self.l1.forward(x)))

        return out
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """

        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers. 
        grad_feed_forward = self.l1.backward(self.activation.backward(self.l2.backward(grad)))

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward

    def step_gd(self,step_size):
 
        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)


@jitclass([
    ('self_attention', SelfAttention.class_type.instance_type),
    ('feed_forward', FeedForward.class_type.instance_type)
])
class TransformerBlock(Layer):
    def __init__(self, d, k, p):
        self.self_attention = SelfAttention(d, k)
        self.feed_forward = FeedForward(d, p, 0.1)
    
    def forward(self, z):
        z_l_half = self.self_attention.forward(z) + z
        z_l = self.feed_forward.forward(z_l_half) # has resnet connection
        return z_l

    def backward(self, grad):
        dL_dz_half = self.feed_forward.backward(grad)
        grad = self.self_attention.backward(dL_dz_half) + dL_dz_half
        return grad

    def step_gd(self, optimizer: Optimizer):
        self.self_attention.step_gd(optimizer)
        self.feed_forward.step_gd(optimizer)