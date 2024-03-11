import numpy as np
from numba import njit
import numba as nb

#hei


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
    b, n = x.shape

    #Making sure that x is an array of integers
    x = x.astype(int)
    x_one_hot = np.zeros((b, m, n))
    x_one_hot[np.arange(b)[:,None],x,np.arange(n)[None,:]] = 1
    return x_one_hot

@njit(inline='always')
def numba_max_axis1(a):
    '''
    Finds the maximum value of each row in a 3D matrix.
    '''
    # Keeps dims
    assert a.ndim == 3
    running = np.zeros((a.shape[0], 1, a.shape[2]))
    for i in range(a.shape[1]):
        running = np.maximum(running, a[:, i:i+1, ...])
    return running

@njit(inline='always')
def numba_mean_axis0(a):
    '''
    Finds the mean of each colomn in a 3D matrix.
    '''
    # Reduces dims
    assert a.ndim == 3
    running = np.zeros((a.shape[1], a.shape[2]))
    for i in range(a.shape[0]):
        running += a[i, ...]
    return running / a.shape[0]

@njit(inline='always')
def numba_mean_axis1(a):
    '''
    Finds the mean of each row in a 3D matrix.
    '''
    # Reduces dims
    running = np.zeros((a.shape[0],))
    for i in range(a.shape[1]):
        running += a[:, i]
    return running / a.shape[1]

@njit
def _jit_onehot(x, m):
    '''
    Onehot function that can be used with numba.
    '''
    b, n = x.shape
    x_one_hot = np.zeros((b, m, n))
    for i in range(b):
        for j in range(n):
            x_one_hot[i, x[i, j], j] = 1
    return x_one_hot

def jit_onehot(x, m):
    x = x.astype(np.int64)
    return _jit_onehot(x, m)


@njit(inline='always')
def numba_mean_bias(grad_bias):
    '''
    Calculates the mean bias of a three dimensional numpy array.
    '''
    out = np.zeros(grad_bias.shape[1])
    for i in range(grad_bias.shape[0]):
        for j in range(grad_bias.shape[1]):
            for k in range(grad_bias.shape[2]):
                out[j] += grad_bias[i, j, k]
    return out

@njit
def make_D_matrix(n):
    '''
    Lower triangular matrix, with -inf under the diagonal, an zeros above
    '''
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
    '''
    A more efficient way to do matrix multiplication than @
    Either A or B or both can have a batch dimension so we have 4 different scenarios

    '''
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