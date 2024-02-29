import gpytorch
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import sys
import os

from GPEC.distances import l2_samples_point, l2_parallel
from GPEC.utils import *

class weighted_geodesic(gpytorch.kernels.Kernel):
    def __init__(self, manifold_samples, geo_matrix, lam = .1, rho = .1, kernel_normalization = True, **kwargs):
        super().__init__(**kwargs)
        self.manifold_samples = manifold_samples
        self.lam = lam
        self.rho = rho
        self.dist_function = l2_samples_point
        self.kernel_normalization = kernel_normalization

        # Calculate Exponential Geodesic Kernel
        self.exp_geo_kernel = utils_torch.exp_kernel_func(geo_matrix.float(), lam = self.lam, q = 1)
        

    def q_weighting(self, x, manifold_samples, rho):
        '''
        Calculate weighted distribution q(m | x, \rho)

        '''
        # get weights to calculate q(m | x, \rho)
        dist = l2_parallel(x, manifold_samples) # n x m
        dist = utils_torch.exp_kernel_func(dist, lam = rho, q = 2) # n x m
        
        # normalization constant Z
        dist_norm = dist.sum(axis = -1) # n-dimensional vector

        # if all weights = 0, set to uniform distribution over manifold samples, i.e. q(m | x, \rho) = p(m)
        zero_idx = torch.where(dist_norm == 0)[0]
        dist[zero_idx,:] = 1
        dist_norm[zero_idx] = dist.shape[1]
        
        dist_norm = dist_norm.unsqueeze(-1).expand(-1,dist.shape[-1]) # n x m
        dist = torch.div(dist, dist_norm) # n x m

        return dist


    def forward(self, x1, x2, diag = False, **kwargs):
        # TODO: have option for when x1 == x2
        # TODO: Use torch cdist instead of l2_parallel
        
        if len(self.batch_shape) == 0:
            batch_size = 0
        else:
            batch_size = self.batch_shape[0]

        if len(x1.shape)>3:
            raise ValueError('Data must be either b x n x d or n x d')
        if len(x1.shape) == 3 and batch_size >1:
            if (torch.equal(x1[0,...], x1[1,...]) is False):
                raise ValueError('Input for each batch dimension should be identical.')

        x1_eq_x2 = torch.equal(x1,x2)
        if batch_size >0:
            # if using batched inputs. We assume that the batched input is identical.
            x1 = x1[0,...]
            x2 = x2[0,...]


        # Calculate weighted distribution q(m | x, \rho)
        q1 = self.q_weighting(x1, self.manifold_samples, self.rho).float()
        if x1_eq_x2:
            q2 = q1
        else:
            q2 = self.q_weighting(x2, self.manifold_samples, self.rho).float()

        # Apply weighting
        output = q1 @ self.exp_geo_kernel.float() @ q2.transpose(1,0)

        if self.kernel_normalization:
            # normalize kernel s.t. k(x,x) = 1
            if x1_eq_x2:
                n1 = n2 = output # n x n
            else:
                n1 = (q1 @ self.exp_geo_kernel @ q1.transpose(1,0)) # n x n
                n2 = (q2 @ self.exp_geo_kernel @ q2.transpose(1,0)) # n x n

            n1 = n1.diag() ** 0.5 # n-dimensional vector
            n2 = n2.diag() ** 0.5 # n-dimensional vector
            normfactor = torch.outer(n1, n2) # n x n

            if torch.div(output, normfactor).isnan().sum() >=1:
                warnings.warn('Divide by zero error when normalizing kernel')

            output = torch.div(output, normfactor)
        
        # output = output.type(dtype=torch.float64)

        if diag:
            output = output.diag()
        
        # Copy kernel over batches
        if batch_size >0:
            if len(output.shape) == 1:
                output = output.unsqueeze(0).expand(batch_size,-1)
            else:
                output = output.unsqueeze(0).expand(batch_size,-1,-1)

        return output.float()

class GPEC(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, manifold_samples, geo_matrix, kernel = 'WEG', lam = 0.1, rho = 0.1, kernel_normalization = True, batch_shape = torch.Size([]), **kwargs):
        super(GPEC, self).__init__(x_train, y_train, likelihood)

        self.batch_shape = batch_shape
        self.mean_module = gpytorch.means.ConstantMean(batch_shape = batch_shape)
        if kernel == 'RBF':
            self.covar_module = (gpytorch.kernels.RBFKernel(batch_shape = batch_shape))
            self.covar_module.lengthscale = lam
        elif kernel == 'WEG':
            self.covar_module = weighted_geodesic(manifold_samples, geo_matrix, lam = lam, rho = rho, kernel_normalization = kernel_normalization, batch_shape = batch_shape)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def train_GPEC(x_train, y_train, manifold_samples = None, geo_matrix = None, kernel = 'WEG', var_list = None, n_iter = 10, lam = 0.1, rho = 0.1, kernel_normalization = True, batch_shape = torch.Size([]), lr = 0.1, optimizer = 'Adam', learn_addn_noise = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if kernel == 'WEG' and (manifold_samples is None or geo_matrix is None):
        raise ValueError('manifold_samples and geo_matrix must be provided when using WEG kernel')

    # print('----------------')
    # print('device: '+ str(device))

    if kernel == 'WEG':
        [x_train, y_train, manifold_samples, geo_matrix] = utils_torch.batch2cuda([x_train, y_train, manifold_samples, geo_matrix])
    else:
        [x_train, y_train] = utils_torch.batch2cuda([x_train, y_train])
    if var_list is not None:
        var_list = utils_torch.auto2cuda(var_list)
    x_train = x_train.float()
    y_train = y_train.float()
    # if manifold_samples is not None: manifold_samples = manifold_samples.float()
    # if geo_matrix is not None: geo_matrix = geo_matrix.float()
    # x_train = x_train.type(dtype=torch.float64)
    # y_train = y_train.type(dtype=torch.float64)


    # likelihood
    if var_list is None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

    else:
        # when using additional variance input (BayesLIME / BayesSHAP)
        var_list = var_list.to(device).float()
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(var_list, learn_additional_noise=learn_addn_noise)

    # initialize model
    model = GPEC(x_train, y_train, likelihood, manifold_samples, geo_matrix, kernel = kernel, lam = lam, rho = rho, kernel_normalization = kernel_normalization, batch_shape = batch_shape)

    # move to gpu
    model = model.to(device)
    likelihood = likelihood.to(device)

    # train model
    model.train()
    likelihood.train()

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
        # print('Optimizer: Adam')
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)  
        # print('Optimizer: SGD')
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pbar = tqdm(range(n_iter))
    for i in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train).sum()
        loss.backward()

        '''
        print('Iter %d/%d - Loss: %.3f' % (
            i + 1, n_iter, loss.item()
        ))
        '''
        optimizer.step()
        pbar.set_postfix({'Loss':loss.item()})
    # print('done!')

    return model, likelihood

def get_pred(x, model, likelihood):

    # move to gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    likelihood = likelihood.to(device)
    x = torch.from_numpy(x)
    # x = x.to(device).type(dtype=torch.float64)
    x = x.to(device).float()
    

    batch_shape = model.batch_shape # get batch shape from model
    if len(batch_shape) == 0:
        batch_size = 0
    else:
        batch_size = batch_shape[0]

    # resize input to batch shape
    if batch_size is None:
        x = x.unsqueeze(0).expand(x.shape[-1], -1, -1)
    else:
        x = x.unsqueeze(0).expand(batch_size, -1, -1)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x))
    return observed_pred
