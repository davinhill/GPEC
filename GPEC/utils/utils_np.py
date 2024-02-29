import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x * (x>0)

def softplus(x):
    return np.log(1+np.exp(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def np_insert(matrix, vector, index):
    '''
    insert vector into matrix (as column) at index
    '''
    matA = matrix[:, :index]
    matB = matrix[:, index:]
    return np.concatenate((matA, vector, matB), axis = 1)

def np_collapse(matrix, index):
    '''
    remove column from matrix
    '''
    matA = matrix[:, :index]
    matB = matrix[:, index+1:]
    return np.concatenate((matA, matB), axis = 1)

def subsample_rows(matrix1, max_rows, matrix2 = None, seed = 0):
    '''
    randomly samples rows of a matrix if the number of rows is greater than max_rows

    args:
        matrix: matrix to be sampled.
        max_rows: maximum number of rows that the matrix should contain
    
    return:
        subsampled matrix
    '''
    n_rows = matrix1.shape[0]
    np.random.seed(seed)
    if n_rows > max_rows:
        idx = np.random.choice(n_rows, size = max_rows, replace = False)
        matrix1 = matrix1[idx,...]
        if matrix2 is not None:
            matrix2 = matrix2[idx,...]
            return (matrix1, matrix2)
        else:
            return matrix1
    else:
        if matrix2 is not None:
            return (matrix1, matrix2)
        else:
            return matrix1

def exp_kernel_func(mat, lam=0.5, q=2):
    '''
    elementwise exp(-lam * mat^q)

    input:
        mat: matrix of distances
        lam: lambda
        q: q

    '''
    return np.exp(-lam * (mat ** q))

def binary_mc2sc_modelwrapper(model):
    '''
    wrapper to convert the 2d output (n x 2) of binary classifiers to 1d (n) by selecting dimension 1 of the output.
    '''
    def wrapper(*args, **kwargs):
        output = model(*args, **kwargs)
        return output[:,1]
    return wrapper

def superpixel_transform_matrix(n_features, superpixel_size):
    '''
    converts a n_features x n_features matrix to superpixel x superpixel matrix by summing adjacent pixels.

    return:
        binary matrix of superpixel x n_features

    '''
    n_superpixels = np.ceil(n_features / superpixel_size).astype('int')
    t_matrix = np.zeros((n_features, n_superpixels)) # matrix to transform feature attributions
    for i in range(n_superpixels):
        grouped_features = np.arange(i * superpixel_size, min(i * superpixel_size + superpixel_size, n_features))
        t_matrix[grouped_features,i] = 1
    return t_matrix
