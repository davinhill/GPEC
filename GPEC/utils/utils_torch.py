import torch
import numpy as np
import os
import sys
import torch.nn.functional as F

#from transformers import PreTrainedTokenizerBase

def tensor2numpy(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()
    return x

def list2cuda(list):
    # adapted from https://github.com/MadryLab/robustness
    array = np.array(list)
    return numpy2cuda(array)

def numpy2cuda(array):
    # function borrowed from https://github.com/MadryLab/robustness
    tensor = torch.from_numpy(array)
    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    # function borrowed from https://github.com/MadryLab/robustness
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def auto2cuda(obj):
    # Checks object type, then calls corresponding function
    if type(obj) == list:
        return list2cuda(obj)
    elif type(obj) == np.ndarray:
        return numpy2cuda(obj)
    elif type(obj) == torch.Tensor:
        return tensor2cuda(obj)
    else:
        raise ValueError('input must be list, np array, or pytorch tensor')

def batch2cuda(list):
    # Input: list of objects to convert. Iterates auto2cuda over list
    output_list = []
    for obj in list:
        output_list.append(auto2cuda(obj))
    return output_list


def idx_to_binary(index_tensor, n_cols = None):
    '''
    Converts vector of indices, where each element of the vector corresponds to a column index in each row of a matrix, into a binary matrix.

    args:
        index_tensor: d-dimensional vector
    returns:
        d x d binary matrix
    '''

    # save input type
    source_type = type(index_tensor)
    if source_type == torch.Tensor:
        device = index_tensor.device
    else:
        device = None

    index_tensor = auto2cuda(index_tensor)

    if n_cols is None:
        n_cols = index_tensor.max().item()+1

    '''
    index_tensor = index_tensor.reshape(-1,1)
    output = tensor2cuda(torch.zeros((index_tensor.shape[0], n_cols), dtype = torch.int32))
    source = tensor2cuda(torch.ones_like(index_tensor, dtype = output.dtype))

    output = output.scatter(dim = 1, index = index_tensor, src = source)
    '''

    output = F.one_hot(index_tensor, num_classes = n_cols)

    # revert type to match input
    if device is not None:
        output = output.to(device)
    if source_type == np.ndarray:
        output = tensor2numpy(output)


    return output



def load_model(path):
    tmp = os.path.dirname(os.path.abspath(path))
    sys.path.append(tmp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = torch.load(path, map_location=device)

    return model

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return min(lr)

def relu2softplus(model, softplus_beta = 1):
    '''
    Given a Pytorch model, convert all ReLU functions to Softplus
    '''
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta = softplus_beta, threshold = 20))
        else:
            relu2softplus(child)

def exp_kernel_func(mat, lam=0.5, q=2, scaling = 0):
    '''
    elementwise exp(-lam * mat^q)

    input:
        mat: matrix of distances
        lam: lambda
        q: q

    '''
    return torch.exp(-lam * (mat ** q) + scaling)

def normalize_l2(mat, dim=1):
    '''
    given a matrix m, normalize all elements such that the L2 norm along dim equals 1

    input:
        mat: matrix
        dim: dimension to normalize

    '''
    norm = torch.norm(mat, dim = dim)
    return torch.div(mat, norm.unsqueeze(dim).expand(mat.shape))

def normalize_01(mat, dim=1):
    '''
    given a matrix m, normalize all elements such that the L2 norm along dim to be within [0,1]

    input:
        mat: matrix
        dim: dimension to normalize

    '''
    min_value = torch.min(mat, dim = dim)[0].unsqueeze(dim).expand(mat.shape)
    max_value = torch.max(mat, dim = dim)[0].unsqueeze(dim).expand(mat.shape)
    norm = max_value - min_value

    return (mat - min_value) / norm


def downsample_images(images, downsample_factor):
    '''
    n x c x h x w matrix. assumes h == w.
    '''

    images = numpy2cuda(images)
    images = images.sum(dim = 1) # sum over channels
    m = torch.nn.AvgPool2d(downsample_factor, stride=downsample_factor, divisor_override=1)
    return tensor2numpy(m(images))