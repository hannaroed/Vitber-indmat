from __future__ import annotations
from abc import ABC, abstractmethod
import functools
import numpy as np
from utils import onehot

class Layer:
    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        self.params = {}

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
        for param in self.params.values():
            optimizer.update(param)



@functools.lru_cache(maxsize=None)
def make_D_matrix(n):
    D = np.zeros((n, n))
    i1, i2 = np.tril_indices(n, -1)
    D[i1,i2] -= np.inf
    return D[None, :, :]


class Matmul(Layer):
    def __init__(self):
        self.prev_A: np.ndarray | None = None
        self.prev_B: np.ndarray | None = None
    
    def forward(self, A, B):
        self.prev_A = A
        self.prev_B = B
        return A @ B
    
    def backward(self, dL_dAB):
        print(dL_dAB.shape, self.prev_B.shape, self.prev_A.shape)
        dL_dA = dL_dAB @ self.prev_B.transpose(0, 2, 1)
        dL_dB = self.prev_A.transpose(0, 2, 1) @ dL_dAB
        print(dL_dA.shape, dL_dB.shape)
        return dL_dA, dL_dB


class Attention(Layer):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
        self.matmul1 = Matmul()
        self.matmul2 = Matmul()

    def forward(self, Q, K, V):
        b, d, n = Q.shape
        D = make_D_matrix(n)

        print(Q.shape, K.shape, np.transpose(Q, axes=(0, 2, 1)).shape)
        qk_prod = self.matmul1.forward(Q.transpose(0, 2, 1), K) / np.sqrt(d)

        A = self.softmax.forward(qk_prod + D)

        VA = self.matmul2.forward(V, A)

        return VA
    
    def backward(self, dL_dVA):
        dL_dV, dL_dA = self.matmul2.backward(dL_dVA)

        dL_dqk_prod = self.softmax.backward(dL_dA)

        dL_dQ, dL_dK = self.matmul1.backward(dL_dqk_prod)
        print(dL_dQ.shape)
        dL_dQ = dL_dQ.transpose(0, 2, 1)

        b, d, n = dL_dQ.shape

        dL_dQ /= np.sqrt(d)
        dL_dK /= np.sqrt(d)

        return dL_dQ, dL_dK, dL_dV


class SelfAttention(Layer):

    def __init__(self, d, k):
        super().__init__()
        self.W_q = LinearLayer(d, k, init_scale=0.1)
        self.W_k = LinearLayer(d, k, init_scale=0.1)
        self.W_v = LinearLayer(d, k, init_scale=0.1)
        self.W_o = LinearLayer(k, d, init_scale=0.1)

        self.prev_A: np.ndarray | None = None

        self.softmax = Softmax()

        self.attention = Attention()


    def forward(self, z):
        b, k, n = z.shape

        Q = self.W_q.forward(z)  # (b, k, n)
        K = self.W_k.forward(z)  # (b, k, n)
        V = self.W_v.forward(z)  # (b, k, n)

        VA = self.attention.forward(Q, K, V)

        out = self.W_o.forward(VA)

        return out


    def backward(self, grad):
        dL_dVA = self.W_o.backward(grad)  # (b, k, n)

        dL_dQ, dL_dK, dL_dV = self.attention.backward(dL_dVA)  # each (b, k, n)

        dL_dz = self.W_q.backward(dL_dQ) + self.W_k.backward(dL_dK) + self.W_v.backward(dL_dV)

        return dL_dz
    
    def step_gd(self, alpha):
        self.W_q.step_gd(alpha)
        self.W_k.step_gd(alpha)
        self.W_v.step_gd(alpha)
        self.W_o.step_gd(alpha)
    


class Softmax(Layer):

    def __init__(self, epsilon: float = 10e-8):
        super().__init__()
        self.epsilon = epsilon

        self.prev_Q: np.ndarray | None = None
        self.prev_P: np.ndarray | None = None
        self.prev_z_l: np.ndarray | None = None
    
    def forward(self, x):
        """Columnwise softmax operation"""
        # x: (batch, d, n)
        self.x = x

        shifted = np.where(np.isneginf(x), x, x - x.max(axis=1, keepdims=True))

        P = np.exp(shifted)
        Q = np.sum(P, axis=0, keepdims=True)

        z_l = P / (Q + self.epsilon)

        self.prev_P = P
        self.prev_Q = Q
        self.prev_z_l = z_l

        return z_l


    def backward(self, grad):
        P, Q, z_l = self.prev_P, self.prev_Q, self.prev_z_l
        
        S = P / (Q * Q + self.epsilon)
        # print(f'{S.shape=}')
        # print(f'{z_l.shape=}')
        # print(f'{(grad * S).shape=}')

        dL_dz = grad * z_l - np.sum(grad * S, axis=1, keepdims=True) * P

        print(f'{dL_dz.shape=}')
        
        return dL_dz


class CrossEntropy(Layer):

    def __init__(self, epsilon = 10e-18):
        super().__init__()
        self.epsilon = epsilon
        self.prev_y_pred: np.ndarray | None = None
        self.prev_y: int | None = None
        

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Cross entropy definition in assignment is wrong"""
        # y_pred: (batch, d, n)
        # y_true: (batch, d)
        # d = embedding dimension = number of classes
        self.prev_y_pred = y_pred
        self.prev_y = y_true

        # out: (batch,)

        batch_index = np.arange(y_pred.shape[0])[:, None]
        seq_index = np.arange(y_pred.shape[2])[None, :]

        out = -np.log(y_pred[batch_index, y_true, seq_index] + self.epsilon).mean(axis=1)

        return out

    def backward(self):
        out = np.zeros_like(self.prev_y_pred)

        hot = 1 / self.prev_y_pred[:, self.prev_y, :]

        out[:, self.prev_y, :] = -hot

        print(f'loss: {out.shape=}')

        return out
    


class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self, input_size, output_size, init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """
        super().__init__()

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w), }}
        

    def forward(self, x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('od,bdn->bon', self.params['w']['w'], x)
        return y
        
    def backward(self, grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        b = grad.shape[0]
        print(f'{grad.shape=}')
        print(f'{self.x.shape=}')

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od', grad, self.x) / b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn', self.params['w']['w'], grad)
    

class TransformerBlock(Layer):
    def __init__(self, d, k, p):
        super().__init__()
        self.self_attention = SelfAttention(d=d, k=k)
        self.feed_forward = FeedForward(d=d, p=p)
    
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


class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        super().__init__()

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


class EmbedPosition(Layer):
    def __init__(self, n_max, m, d, init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max) * init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp": {'w':self.w, 'd':None}}

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
        print(X.shape)
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
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)




class FeedForward(Layer):
    def __init__(self,d, p, init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        self.x: np.ndarray | None = None

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)


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
        print(f'{x.shape=}')

        self.x = x

        out = x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
        print(f'{out.shape=}')
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



class Optimizer(ABC):
    @abstractmethod
    def update(self, parameters: dict[str, np.ndarray]) -> None:
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, alpha: float = 0.01, epsilon: float = 10e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.epsilon = epsilon
        self.step = 0
    
    def update(self, parameters: dict[str, np.ndarray]):
        """
        Takes in gradients, parameters, and previous moments and returns the update step and both .
        """
        w, grad, m_prev, v_prev = parameters['w'], parameters['d'], parameters.get('m', None), parameters.get('v', None)
        m_prev = np.zeros_like(w) if m_prev is None else m_prev
        v_prev = np.zeros_like(w) if v_prev is None else v_prev

        # Use m_prev and v_prev to find m and v

        self.step += 1

        m = self.beta1 * m_prev + (1-self.beta1) * grad
        v = self.beta2 * v_prev + (1-self.beta2)*(grad*grad)
        m_hat = (1/(1-self.beta1**self.step))*m
        v_hat = (1/(1-self.beta1**self.step))*v
        step = self.alpha * (m_hat / (np.sqrt(v_hat) + self.epsilon))

        parameters['w'] -= step
        parameters['m'] = m
        parameters['v'] = v


