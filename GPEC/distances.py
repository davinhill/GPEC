import pdb
import numpy as np
import torch



class geodistance_toy_1d():
    '''
    geodesic distance on a 1d manifold embedded in R2
    '''
    def __init__(self, decision_boundary, n_steps = 50):
        self.decision_boundary = decision_boundary
        self.n_steps = n_steps
    def __call__(self, x1, x2):
        if type(x1) == np.ndarray:
            if x1.shape[0] == 2:
                x1 = x1[0]
                x2 = x2[0]
        # if x1 <= x2:
        #     x_samples = np.arange(x1, x2+self.step, self.step).reshape(-1,1)
        # else:
        #     x_samples = np.arange(x2, x1+self.step, self.step).reshape(-1,1)
        x_samples = np.linspace(x1, x2, self.n_steps, endpoint = True).reshape(-1,1)
        y_samples = self.decision_boundary(x_samples)
        samples = np.hstack((x_samples,y_samples))
        dist = np.linalg.norm(samples[1:,:] - samples[:-1,:], ord = 2, axis = 1)
        return dist.sum()

class eucdistance_toy_1d():
    '''
    geodesic distance on a 1d manifold embedded in R2
    '''
    def __init__(self, decision_boundary):
        self.decision_boundary = decision_boundary
    def __call__(self, x1, x2):
        if type(x1) == np.ndarray:
            if x1.shape[0] == 2:
                x1 = x1[0]
                x2 = x2[0]
        x_samples = np.linspace(x1, x2, 2, endpoint = True).reshape(-1,1)
        y_samples = self.decision_boundary(x_samples)
        samples = np.hstack((x_samples,y_samples))
        dist = np.linalg.norm(samples[0,:] - samples[1,:], ord = 2)
        return dist.sum()


def geomatrix(samples, distance_func):
    '''
    np array of (scalar) samples
    '''
    K = np.zeros((samples.shape[0], samples.shape[0]))
    for i, x in enumerate(samples):
        for j in np.arange(i, samples.shape[0]):
            if i == j:
                K[i,j] == 0
            else:
                y = samples[j]
                K[i,j] = distance_func(x,y)
    
    return K + K.transpose()

def l2_samples_point(x, samples):
    '''
    calculate l2 distances between samples (nxd matrix) and a given point (1xd matrix)
    '''
    convert = False
    if type(x) == np.ndarray:
        x, samples = torch.tensor(x), torch.tensor(samples)
        convert = True

    point = x.expand(samples.shape[0], -1)
    l2_dist = torch.norm(samples - point, p = 2, dim = 1) ** 2

    if convert:
        l2_dist = l2_dist.numpy()

    return l2_dist

def torch_dotlastdim(mat1, mat2):
    # take dot product over last dimensions over two tensors
    mat1 = mat1.unsqueeze(-1)
    mat2 = mat2.unsqueeze(-2)
    return (mat1 @ mat2)[...,0,0]



def l2_parallel(x, samples):
    '''
    calculate l2 distances between manifold samples (mxd matrix) and a given points (nxd matrix).
    Result is a nxm matrix.
    '''
    convert = False
    if type(x) == np.ndarray:
        x, samples = torch.tensor(x), torch.tensor(samples)
        convert = True
    else:
        device = x.device
        x, samples = x.cpu(), samples.cpu()

    m = samples.shape[0]
    n = x.shape[0]

    x = x.unsqueeze(1).expand(-1, m, -1)  # n x m x d
    samples = samples.unsqueeze(0).expand(n, -1, -1) # n x m x d

    mat = x - samples  # n x m x d
    output = torch.norm(mat, p = 2, dim = -1)
    
    # output = torch_dotlastdim(mat, mat) # dot product over last dimension

    if convert:
        output = output.numpy()
    else:
        output = output.to(device)

    return output # n x m
