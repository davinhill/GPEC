import numpy as np
import os
import warnings
try:
    import networkx as nx
except:
    warnings.warn('Networkx is not installed')
try:
    import matplotlib.pyplot as plt
except:
    warnings.warn('Matplotlib is not installed')
import itertools
from tqdm import tqdm
import sys
import torch
import torch.nn.functional as F
import time

from GPEC import riemannian_tree
from GPEC.utils import *


def toy_cosine_manifold(x):
    '''
    DO NOT USE -- outdated. 
    Decision boundary for cosine toy model.
    
    '''
    x = np.sign(x) * np.maximum(np.abs(x), np.zeros_like(x) + 1e-6) # numerical stability

    return 2*np.cos(10/x)


def plt_gridsearch_orig(xx,yy,probs, decision_threshold=0.5, n_neighbors = 10, C = 1e4):
    '''
    DO NOT USE -- outdated. 
    
    '''
    cs = plt.contour(xx, yy, probs, levels=[decision_threshold], cmap = "Greys", vmin=-0.6, vmax=.1, linewidths=3)

    ##############################3
    path_list = cs.collections[0].get_paths()
    print('paths: %s ' % str(len(path_list)))

    G = nx.Graph()
    sample_list = []
    for i,path in enumerate(path_list):
        manifold_samples = path.vertices

        # skip if too few samples
        if manifold_samples.shape[0] < n_neighbors: continue

        rTree = RiemannianTree()
        tmp = rTree.create_riemannian_graph(manifold_samples, n_neighbors=n_neighbors)
        sample_list.append(manifold_samples)
        last_node = len(G)-1
        G = nx.disjoint_union(G, tmp)
        edge_attr = {'weight_euclidean': float(1/C),
                    'distance_euclidean': float(C)}
        if i != 0:
            G.add_edge(last_node, last_node+1, **edge_attr)

    manifold_samples = np.concatenate(sample_list, axis = 0)
    geo_matrix = np.zeros((len(G),len(G)))
    for i, (src, dic) in enumerate(nx.shortest_path_length(G, weight = 'distance_euclidean')):
        for j in range(i, geo_matrix.shape[0]):
            geo_matrix[src,j] = dic[j]

    geo_matrix = geo_matrix + geo_matrix.transpose()

    return geo_matrix, manifold_samples

def create_grid(data, gridsize = 100):
    '''
    create grid of same dimension and range of input dataset/

    args:
        data: (np matrix) size n x d
        gridsize: (int) number of grid elements in each dimension
    return:
        grid: (np matrix) size gridsize^d x d
        xx_list: d-length list of gridsize x gridsize np matrices
    '''
    
    linspace_list = []
    for feat_idx in range(data.shape[1]):
        feat_min = data[:,feat_idx].min()
        feat_max = data[:, feat_idx].max()
        linspace_list.append(np.linspace(feat_min, feat_max, gridsize, dtype = data.dtype))
        
    xx_list = np.meshgrid(*linspace_list)
    xx_list = [xx.transpose() for xx in xx_list]
    ravel_list = [xx.ravel() for xx in xx_list]
    grid = np.vstack(ravel_list).transpose()

    return xx_list, grid

def create_grid_db(data, gridsize = 10):
    '''
    create grid of same dimension and range of input dataset.
    
    Same as create_grid, but omits xx_list calculation for efficiency.

    args:
        data: (np matrix) size n x d
        gridsize: (int) number of grid elements in each dimension
    return:
        grid: (np matrix) size gridsize^d x d
    '''
    
    linspace_list = []
    for feat_idx in range(data.shape[1]):
        feat_min = data[:,feat_idx].min()
        feat_max = data[:, feat_idx].max()
        linspace_list.append(np.linspace(feat_min, feat_max, gridsize, dtype = data.dtype))
        
    xx_list = np.meshgrid(*linspace_list)
    grid = np.vstack(tuple(map(np.ravel, xx_list)))

    return grid.transpose()



def advsamples_xgboost(data, labels, n_samples, xgb_model):
    '''
    Uses adversarial attacks to generate samples close to the decision boundary. Using these 'close' samples speeds up the subsequent binary search.

    args:
        data: (np matrix) size n x d
        gridsize: (int) number of grid elements in each dimension
    return:
        grid: (np matrix) size gridsize^d x d
        xx_list: d-length list of gridsize x gridsize np matrices
    '''

    sys.path.append('./GPEC/utils/')
    from z_ART_XGBoostClassifier import XGBoostClassifier
    from art.attacks.evasion import ZooAttack
    from art.utils import to_categorical
    import xgboost
    
    pred = xgb_model.predict(xgboost.DMatrix(data))
    pred0 = np.where(pred < 0.5)[0]
    pred1 = np.where(pred >= 0.5)[0]
    # draw equal number of samples from each class

    x_list = []
    y_list = []
    for (x,y) in [(data[pred0,...],labels[pred0]), (data[pred1,...], labels[pred1])]:
        x, y = utils_np.subsample_rows(matrix1 = x, matrix2 = y, max_rows = n_samples//2)
        x_list.append(x)
        y_list.append(y)
    x_seed = np.concatenate(x_list, axis = 0)
    y_seed = np.concatenate(y_list, axis = 0)
    
    classifier = XGBoostClassifier(model=xgb_model, nb_features=data.shape[1], nb_classes=2)
    attack = ZooAttack(
        classifier=classifier,
        confidence=0.0,
        targeted=False,
        learning_rate=0.1,
        max_iter=2000,
        #max_iter=50,
        binary_search_steps=5,
        initial_const=1e-2,
        abort_early=True,
        use_resize=False,
        use_importance=True,
        nb_parallel=10,
        batch_size=1,
        variable_h=0.5,
    )
    x_adv = attack.generate(x=x_seed, y=to_categorical(y_seed))
    pred = (classifier.predict(x_adv)[:,1] >= 0.5)*1
    # adv_accy = metrics.accuracy_score(y_test, pred>=0.5)

    return x_adv

def advsamples_pytorch(model, tr_data_im, tr_loader, args):
    '''
    Uses adversarial attacks to generate samples close to the decision boundary. Using these 'close' samples speeds up the subsequent binary search.

    args:
        model: pytorch model
        x_train: training set (n x d)

    returns:
        dictionary with adversarial samples from each class.
    '''
    
    from GPEC.adv_boundary import generate_adverserial_for_class_wrapper

    AdvDict = generate_adverserial_for_class_wrapper(args, model, tr_data_im, tr_loader)

    return AdvDict

def advsamples_pytorch_batched(model, tr_loader, args):
    '''
    Uses adversarial attacks to generate samples close to the decision boundary. Using these 'close' samples speeds up the subsequent binary search.

    args:
        model: pytorch model
        x_train: training set (n x d)

    returns:
        dictionary with adversarial samples from each class.
    '''
    from adv_boundary_batched import generate_adverserial_for_class_wrapper

    AdvDict = generate_adverserial_for_class_wrapper(args, model, tr_loader)

    return AdvDict


def sampledb_func(data, model, gridsize=1e3):
    '''
    returns samples from the decision boundary assuming we have closed form solution for the decision boundary.

    args:
        data: (np matrix) size n x d
        model: model must have a model.db(x) function that outputs 2d manifold samples

    return:
        manifold samples (np matrix)
    '''

    xmin = data[:,0].min()
    xmax = data[:,0].max()
    int = (xmax - xmin) / gridsize
    x = np.arange(xmin, xmax, int)

    return model.db(x)

def sampledb_plt(xx,yy,probs, decision_threshold = 0.5):
    '''
    returns samples from the decision boundary using matplotlib contours.
    '''
    probs = probs.reshape(xx.shape)
    cs = plt.contour(xx, yy, probs, levels=[decision_threshold], cmap = "Greys", vmin=-0.6, vmax=.1, linewidths=3)
    path_list = cs.collections[0].get_paths()
    print('paths: %s ' % str(len(path_list)))

    vertex_list = [path.vertices for path in path_list]
    manifold_samples = np.concatenate(vertex_list, axis = 0)
    return manifold_samples


def sampledb_DBPS_binary(grid, model, decision_threshold = 0.5, n_samples_per_class = 100, batch_size = 1024, max_iter = 1e5, early_stopping = 10, tol = 5e-2, prob_tol_filter = None, max_samples = None):
    '''
    returns samples from the decision boundary using binary search, adapted from DBPS:

    Zhiyong Yan and Congfu Xu, "Using decision boundary to analyze classifiers," 2008 3rd International Conference on Intelligent System and Knowledge Engineering, 2008, pp. 302-307, doi: 10.1109/ISKE.2008.4730945.
    '''

    if len(model(grid).shape)>1: model = utils_np.binary_mc2sc_modelwrapper(model)

    probs = model(grid)
    
    if prob_tol_filter is not None:
        # filter on samples close to decision boundary
        filter_idx = np.where(np.abs(probs - decision_threshold) <= prob_tol_filter)[0]
        probs = probs[filter_idx]

    # separate samples by predicted class
    c1_idx = np.where(probs >= decision_threshold)[0] # samples from class 1
    c2_idx = np.where(probs < decision_threshold)[0] # samples from class 2

    # filter number of samples
    if c1_idx.shape[0] > n_samples_per_class:
        c1_idx = np.random.choice(c1_idx, size = n_samples_per_class, replace = False)
    if c2_idx.shape[0] > n_samples_per_class:
        c2_idx = np.random.choice(c2_idx, size = n_samples_per_class, replace = False)

    sample_idx = np.array(list(itertools.product(c1_idx, c2_idx))) # s x 2 where s = n_samples_per_class^2
    sample_idx = np.concatenate((sample_idx, np.zeros((sample_idx.shape[0],4))), axis = 1).astype('int32') # s x 4
    # dim 2 = binary indicator for convergence
    # dim 3 = number of iterations on sample
    # dim 4 = binary indicator, converge or max iter reached
    # dim 5 = number of iterations with same gap


    grid_s1 = grid[sample_idx[:,0],:] # Sample 1
    grid_s2 = grid[sample_idx[:,1],:] # Sample 2
    gap_list = np.zeros(sample_idx.shape[0]) # keep track of gap
    gap_list_previous = np.zeros(sample_idx.shape[0])-1 # keep track of previous iteration gap
    converge_list = []

    zero_idx = np.where(sample_idx[:,4] == 0)[0]
    n_remaining = len(zero_idx)

    # loop until all samples have converged
    with tqdm(total = sample_idx.shape[0]) as pbar:
        while(n_remaining>0):
            batch_idx = zero_idx[:(batch_size // 3)] # get indices for current batch
            sample_idx[batch_idx, 3] += 1 # record attempt

            # get samples
            sample1 = grid_s1[batch_idx, :]
            sample2 = grid_s2[batch_idx, :]

            midpoint = (sample1 + sample2) / 2

            # get model output
            test_points = np.concatenate((sample1, sample2, midpoint), axis = 0)
            model_output = model(test_points)
            pred_s1 = (model_output[:sample1.shape[0]] >= decision_threshold) * 1
            pred_s2 = (model_output[sample1.shape[0]:(sample1.shape[0] + sample2.shape[0])] >= decision_threshold) * 1
            output_midpoint = model_output[(sample1.shape[0] + sample2.shape[0]):]
            pred_midpoint = (output_midpoint >= decision_threshold) *1

            
            # Check if samples have converged
            gap = np.abs(output_midpoint - decision_threshold)
            convergence_idx_within_batch= np.where(gap <= tol)[0]
            convergence_idx = batch_idx[convergence_idx_within_batch]
            sample_idx[convergence_idx,2] = 1 # record convergence
            sample_idx[convergence_idx,4] = 1 # record convergence

            converge_list.append(midpoint[convergence_idx_within_batch,:])
            gap_list[convergence_idx] = gap[convergence_idx_within_batch] # save midpoint output

            # Check if max iter has been reached
            timeout_idx_within_batch= np.where(sample_idx[batch_idx, 3] >= max_iter)[0]
            timeout_idx = batch_idx[timeout_idx_within_batch]
            sample_idx[timeout_idx,4] = 1 # flag for stopping
            gap_list[timeout_idx] = gap[timeout_idx_within_batch] # save midpoint output

            # Check early stopping criteria
            gap_diff_flag = (gap_list_previous[batch_idx] == gap)*1 # Check if current gap is different from previous
            sample_idx[batch_idx,5] = sample_idx[batch_idx,5] * gap_diff_flag + gap_diff_flag # update counter
            early_idx_within_batch = np.where(sample_idx[batch_idx, 5] >= early_stopping)[0] # check number of iterations with same gap
            early_idx = batch_idx[early_idx_within_batch]
            sample_idx[early_idx,4] = 1 # flag for stopping
            gap_list[early_idx] = gap[early_idx_within_batch] # save midpoint output
            gap_list_previous[batch_idx] = gap # record previous gap

            # update sample 1. if pred_midpoint == pred_s1, set s1 = midpoint. Otherwise, keep s1.
            class_check = (pred_midpoint == pred_s1) * 1
            class_check = class_check.reshape(-1,1).repeat(midpoint.shape[1], axis = 1)
            grid_s1[batch_idx,:] = grid_s1[batch_idx,:] * (1-class_check) + midpoint * class_check

            # update sample 2
            class_check = (pred_midpoint == pred_s2) * 1
            class_check = class_check.reshape(-1,1).repeat(midpoint.shape[1], axis = 1)
            grid_s2[batch_idx,:] = grid_s2[batch_idx,:] * (1-class_check) + midpoint * class_check

            zero_idx = np.where(sample_idx[:,4] == 0)[0] # identify samples which have not converged
            pbar.update(n_remaining - len(zero_idx))
            n_remaining = len(zero_idx)
    
    not_converged_idx = np.where(gap_list > tol)[0]
    if len(not_converged_idx) == 0:
        mean_gap = 0
        max_gap = 0
    else:
        mean_gap = gap_list[not_converged_idx].mean()
        max_gap = gap_list[not_converged_idx].max()
    manifold_samples = np.concatenate(converge_list, axis = 0)

    print("# Samples Not Converged: %s" % str(len(not_converged_idx)))
    print("Avg Gap: %s" % str(mean_gap))
    print("Max Gap: %s" % str(max_gap))

    if max_samples is not None and manifold_samples.shape[0] > max_samples:
        idx = np.random.choice(manifold_samples.shape[0], size = max_samples, replace = False)
        manifold_samples = manifold_samples[idx,...]
        print('Samples limited to: %s' % str(max_samples))
    return manifold_samples

def sampledb_DBPS_multiclass(samples, target_class, model, n_samples_per_class = 100, batch_size = 1024, max_iter = 1e3, early_stopping = 10, tol = 1e-2):
    '''
    returns samples from the one-vs-all decision boundary using binary search.

    args:
        samples: (n x d pytorch tensor) data samples
        target_class: (int) target class in one-vs-all prediction
        model: pytorch model

    '''

    samples = samples.cpu()
    logits = model(utils_torch.tensor2cuda(samples))
    preds = torch.argmax(logits, dim=1)
    preds = utils_torch.tensor2numpy(preds)
    
    # separate samples by predicted class
    c1_idx = np.where(preds == target_class)[0] # samples from target class
    c2_idx = np.where(preds != target_class)[0] # samples from all other classes

    # filter number of samples
    if n_samples_per_class < c1_idx.shape[0]:
        c1_idx = np.random.choice(c1_idx, size = n_samples_per_class, replace = False)
    if n_samples_per_class < c2_idx.shape[0]:
        c2_idx = np.random.choice(c2_idx, size = n_samples_per_class, replace = False)

    sample_idx = np.array(list(itertools.product(c1_idx, c2_idx))) # s x 2 where s = n_samples_per_class^2
    sample_idx = np.concatenate((sample_idx, np.zeros((sample_idx.shape[0],4))), axis = 1).astype('int32') # s x 4
    # dim 2 = binary indicator for convergence
    # dim 3 = number of iterations on sample
    # dim 4 = binary indicator, converge or max iter reached
    # dim 5 = number of iterations with same gap


    grid_s1 = samples[sample_idx[:,0],...] # Sample 1
    grid_s2 = samples[sample_idx[:,1],...] # Sample 2
    gap_list = np.zeros(sample_idx.shape[0]) # keep track of gap
    gap_list_previous = np.zeros(sample_idx.shape[0])-1 # keep track of previous iteration gap
    converge_list = []

    zero_idx = np.where(sample_idx[:,4] == 0)[0]
    n_remaining = len(zero_idx)

    # loop until all samples have converged
    with tqdm(total = sample_idx.shape[0]) as pbar:
        while(n_remaining>0):
            batch_idx = zero_idx[:(batch_size // 3)] # get indices for current batch
            sample_idx[batch_idx, 3] += 1 # record attempt

            # get samples
            sample1 = grid_s1[batch_idx, ...]
            sample2 = grid_s2[batch_idx, ...]

            # concatenate samples
            midpoint = (sample1 + sample2) / 2
            test_points = np.concatenate((sample1, sample2, midpoint), axis = 0)

            # move to gpu
            # TODO: convert algorithm to pytorch
            test_points = utils_torch.numpy2cuda(test_points)

            # get model output
            model_output = model(test_points)
            model_probs = F.softmax(model_output, dim = 1)

            # move to cpu
            model_output, model_probs = utils_torch.tensor2numpy(model_output), utils_torch.tensor2numpy(model_probs)

            # get predictions
            pred_s1 = model_output[:sample1.shape[0]].argmax(axis = 1)
            pred_s2 = model_output[sample1.shape[0]:(sample1.shape[0] + sample2.shape[0]), :].argmax(axis = 1)
            
            # midpoint calculations
            output_midpoint = model_output[(sample1.shape[0] + sample2.shape[0]):,:]
            probs_midpoint = model_probs[(sample1.shape[0] + sample2.shape[0]):,:]
            ranked_preds_midpoint = np.argsort(output_midpoint, axis = 1)
            pred1_midpoint = ranked_preds_midpoint[:,-1] # top prediction
            pred2_midpoint = ranked_preds_midpoint[:,-2] # 2nd highest prediction

            prob1_midpoint = np.multiply(probs_midpoint, utils_torch.idx_to_binary(pred1_midpoint, model_output.shape[1])).sum(axis = 1) # predicted probability of top prediction
            prob2_midpoint = np.multiply(probs_midpoint, utils_torch.idx_to_binary(pred2_midpoint, model_output.shape[1])).sum(axis = 1) # predicted probability of top prediction

            probtarget_midpoint = probs_midpoint[:,target_class] #  # predicted probability of target prediction

            gap = np.multiply((pred1_midpoint == target_class) * 1, np.abs(probtarget_midpoint - prob2_midpoint)) # if the midpoint prediction is the target class, then the gap is between pred1 and pred2
            gap += np.multiply((pred1_midpoint != target_class) * 1, np.abs(probtarget_midpoint - prob1_midpoint)) # if the midpoint prediction is NOT in target class, then the gap is between the target class and the predicted class


            
            # Check if samples have converged
            convergence_idx_within_batch= np.where(gap <= tol)[0]
            convergence_idx = batch_idx[convergence_idx_within_batch]
            sample_idx[convergence_idx,2] = 1 # record convergence
            sample_idx[convergence_idx,4] = 1 # record convergence
            converge_list.append(midpoint[convergence_idx_within_batch,:])
            gap_list[convergence_idx] = gap[convergence_idx_within_batch] # save midpoint output

            # Check if max iter has been reached
            timeout_idx_within_batch= np.where(sample_idx[batch_idx, 3] >= max_iter)[0]
            timeout_idx = batch_idx[timeout_idx_within_batch]
            sample_idx[timeout_idx,4] = 1 # flag for stopping
            gap_list[timeout_idx] = gap[timeout_idx_within_batch] # save midpoint output

            # Check early stopping criteria
            gap_diff_flag = (gap_list_previous[batch_idx] == gap)*1 # Check if current gap is different from previous
            sample_idx[batch_idx,5] = sample_idx[batch_idx,5] * gap_diff_flag + gap_diff_flag # update counter
            early_idx_within_batch = np.where(sample_idx[batch_idx, 5] >= early_stopping)[0] # check number of iterations with same gap
            early_idx = batch_idx[early_idx_within_batch]
            sample_idx[early_idx,4] = 1 # flag for stopping
            gap_list[early_idx] = gap[early_idx_within_batch] # save midpoint output
            gap_list_previous[batch_idx] = gap # record previous gap

            # update sample 1. if pred_midpoint == pred_s1, set s1 = midpoint. Otherwise, keep s1.
            class_check = (pred1_midpoint == pred_s1) * 1
            # reshape class check
            tmp = np.ones(len(midpoint.size()), dtype = np.int16).tolist()
            tmp[0] = len(class_check)
            class_check = torch.from_numpy(class_check.reshape(tmp)).expand(midpoint.size())
            grid_s1[batch_idx,...] = (grid_s1[batch_idx,...] * (1-class_check) + midpoint * class_check).type(dtype = grid_s1.dtype)

            # update sample 2
            class_check = (pred1_midpoint == pred_s2) * 1
            tmp = np.ones(len(midpoint.size()), dtype = np.int16).tolist()
            tmp[0] = len(class_check)
            class_check = torch.from_numpy(class_check.reshape(tmp)).expand(midpoint.size())
            grid_s2[batch_idx,...] = (grid_s2[batch_idx,...] * (1-class_check) + midpoint * class_check).type(dtype=grid_s2.dtype)

            zero_idx = np.where(sample_idx[:,4] == 0)[0] # identify samples which have not converged
            pbar.update(n_remaining - len(zero_idx))
            n_remaining = len(zero_idx)
    
    not_converged_idx = np.where(gap_list > tol)[0]
    mean_gap = gap_list[not_converged_idx].mean()
    max_gap = gap_list[not_converged_idx].max()
    manifold_samples = np.concatenate(converge_list, axis = 0)

    print("# Samples Not Converged: %s" % str(len(not_converged_idx)))
    print("Avg Gap: %s" % str(mean_gap))
    print("Max Gap: %s" % str(max_gap))

    return manifold_samples


def geo_kernel_matrix(manifold_samples, n_neighbors = 5, C = 1e10):
    '''
    given samples from decision boundary, returns matrix of geodesic distances.

    input:
        manifold_samples:
        n_neighbors: knn parameter for Riemannian Tree algorithm.

    return: n x n np matrix of geodesic distances
    '''


    # Create Riemannian Tree Object
    manifold_samples = manifold_samples.reshape(manifold_samples.shape[0],-1)
    rTree = riemannian_tree.RiemannianTree()
    G = rTree.create_riemannian_graph(manifold_samples, n_neighbors=n_neighbors)

    # Ensure all disjoint subgraphs are connected
    subgraph_list = [G.subgraph(c) for c in nx.connected_components(G)]
    H = nx.Graph()
    H = nx.disjoint_union(H, subgraph_list[0])
    for i in tqdm(range(1,len(subgraph_list)), desc = 'connect_disjoint:'):
        last_node = len(H)-1
        H = nx.disjoint_union(H, subgraph_list[i])
        edge_attr = {'weight_euclidean': float(1/C),
                    'distance_euclidean': float(C)}
        H.add_edge(last_node, last_node+1, **edge_attr)

    # Define geodesic matrix
    geo_matrix = np.zeros((len(H),len(H)))
    for i, (src, dic) in tqdm(enumerate(nx.shortest_path_length(H, weight = 'distance_euclidean')), desc='create_kernel', total = geo_matrix.shape[0]):
        for j in range(i, geo_matrix.shape[0]):
            geo_matrix[src,j] = dic[j]
    geo_matrix = geo_matrix + geo_matrix.transpose()

    return geo_matrix

def gpec_db_nn(model, x, y, dataloader, nn_params, save = False, save_path = './'):
    time_start = time.time()
    # get filtered samples
    seed_samples = advsamples_pytorch(model, utils_torch.tensor2cuda(x), dataloader, nn_params)
    # Do I calculate this over classes?
    
    # number of classes
    #classes = torch.unique(tr_model_labels).detach().cpu().numpy()
    classes = list(seed_samples.keys())

    manifold_samples_dict = {}
    geo_matrix_dict = {}
    for target_class in classes:
        # concatenate all adversarial samples for a given class (both targeted and untargeted)
        seed_samples_tmp = torch.concat([seed_samples[target_class][class_j][1] for class_j in list(seed_samples[target_class].keys())], dim = 0) 
        manifold_samples = sampledb_DBPS_multiclass(seed_samples_tmp, target_class = target_class, model = model, n_samples_per_class = 100, batch_size = 4096)
        geo_matrix = geo_kernel_matrix(manifold_samples)

        manifold_samples_dict[target_class] = manifold_samples
        geo_matrix_dict[target_class] = geo_matrix



    output = {
        'manifold_samples': manifold_samples_dict,
        'geo_matrix': geo_matrix_dict,
        'time': time.time() - time_start
    }

    if save: utils_io.save_dict(output, save_path)

    return output

def gpec_db_nn_batched(model, dataloader, nn_params, save = False, save_path = './', save_path_advseeds = None, load_advseeds = False, save_intermediate = False, save_dir_geodesic= None):
    '''
    args:
        y: labels for samples x. Only required if using 
    '''
    time_start = time.time()
    # get filtered samples
    if load_advseeds:
        tmp = utils_io.load_dict(save_path_advseeds)
        seed_samples = tmp['seed_samples']
        time_advseeds = tmp['time']
    else:
        seed_samples = advsamples_pytorch(model, dataloader, nn_params)
        time_advseeds = time.time() - time_start
        if save_path_advseeds is not None:
            tmp = {
                'seed_samples': seed_samples,
                'time': time_advseeds,
            }
            utils_io.save_dict(tmp, save_path_advseeds)
    
    print(save_path_advseeds)
    print('Done!')
    time_start = time.time()
    # number of classes
    #classes = torch.unique(tr_model_labels).detach().cpu().numpy()
    classes = list(seed_samples.keys())

    manifold_samples_dict = {}
    geo_matrix_dict = {}
    for target_class in classes:
        path = save_dir_geodesic + '/db_%s.pkl' % target_class
        # Check if the class has already been calculated
        try:
            utils_io.load_dict(path)
            continue
        except:
            pass

        if save_intermediate:
            print('----------------')
            print('Class: %s' % str(target_class))
            time_start = time.time()
            manifold_samples_dict = {}
            geo_matrix_dict = {}
        # Manifold Samples
        # concatenate all adversarial samples for a given class (both targeted and untargeted)
        seed_samples_tmp = torch.concat([seed_samples[target_class][class_j][1] for class_j in list(seed_samples[target_class].keys())], dim = 0) 
        manifold_samples = sampledb_DBPS_multiclass(seed_samples_tmp, target_class = target_class, model = model, n_samples_per_class = 300, batch_size = 128, max_iter = 1e4)

        # Geodesic Matrix
        geo_matrix = geo_kernel_matrix(manifold_samples)

        manifold_samples_dict[target_class] = manifold_samples
        geo_matrix_dict[target_class] = geo_matrix

        if save_intermediate:
            output = {
                'manifold_samples': manifold_samples_dict,
                'geo_matrix': geo_matrix_dict,
                'time_geomatrix': time.time() - time_start,
                'time_advseeds': time_advseeds / len(classes)
            }
            utils_io.save_dict(output, path)
            del output, manifold_samples, geo_matrix, manifold_samples_dict, geo_matrix_dict  # free up memory


    if not save_intermediate:
        time_geomatrix = time.time() - time_start
        output = {
            'manifold_samples': manifold_samples_dict,
            'geo_matrix': geo_matrix_dict,
            'time': time_geomatrix + time_advseeds
        }

        if save: utils_io.save_dict(output, save_path)
    sys.exit()

    return output
    
