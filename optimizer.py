import numpy as np
import numba as nb
from numba.experimental import jitclass

class Optimizer:
    '''
    A parent class for optimizers, gjør koden mer generell.
    Makes it possible to implement several optimizealgorithms.

    '''
    def update(self, parameters: dict[str, np.ndarray]) -> None:
        raise NotImplementedError


@jitclass([
    ('beta1', nb.float64),
    ('beta2', nb.float64),
    ('alpha', nb.float64),
    ('epsilon', nb.float64),
    ('step', nb.int64),
])
class Adam(Optimizer):
    '''
    Optimize algorithm
    '''
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, alpha: float = 3e-4, epsilon: float = 10e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.epsilon = epsilon
        self.step = 0
    
    def reset(self):
        self.step = 0
    
    def update(self, parameter: dict[str, np.ndarray]):
        """
        Takes in gradients, parameters, and previous moments and returns the update step and both.

        """
        w, grad, m_prev, v_prev = parameter['w'], parameter['d'], parameter.get('m', None), parameter.get('v', None)
        m_prev = np.zeros_like(w) if m_prev is None else m_prev
        v_prev = np.zeros_like(w) if v_prev is None else v_prev

        # Use m_prev and v_prev to find m and v

        self.step += 1

        m = self.beta1 * m_prev + (1 - self.beta1) * grad
        v = self.beta2 * v_prev + (1 - self.beta2) * (grad ** 2)

        m_hat = (1 / (1 - self.beta1**self.step)) * m
        v_hat = (1 / (1 - self.beta1**self.step)) * v

        step = self.alpha * (m_hat / (np.sqrt(v_hat) + self.epsilon))

        parameter['w'] -= step
        parameter['m'] = m
        parameter['v'] = v

       