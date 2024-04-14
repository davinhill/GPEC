
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from skimage.segmentation import slic

import sys
sys.path.append('../../../../')
from GPEC.utils.utils_torch import tensor2cuda, numpy2cuda, tensor2numpy

class eval():
    def __init__(self, model, binary = False, baseline = 'zero'):
        self.model = model
        self.model.eval()


        if type(baseline) == float or type(baseline) == int:
            self.baseline_scaling = baseline
        else:
            self.baseline_scaling = 0

        self.shapley_baseline = baseline
        if self.shapley_baseline == 'mean':
            print('Not Implemented')
            sys.exit()
        self.baseline = None
        self.binary = binary
        self.j = None

    def init_baseline(self, x, num_superpixels, sp_mapping, j = None, fixed_present = True, label = None, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        args:
            x: single sample. numpy array. 1 x c x h x w
            sp_mapping: superpixel to pixel decoder function
        '''

        self.j = j
        self.fixed_present = fixed_present

        x = numpy2cuda(x)
        _, self.c, self.h, self.w = x.shape
        self.x_baseline = x

        # Superpixel mapping
        self.sp_mapping = sp_mapping
        
        # Calculate superpixel map for current sample
        _, self.segment_mask = self.sp_mapping(torch.ones((1, num_superpixels)), x_orig = x)

        if label is None:
            if self.binary:
                self.baseline = torch.sigmoid(self.model(x))
            else:
                self.baseline = self.model(x).argmax(dim = 1)
        else:  
            self.baseline = torch.tensor([int(label)]) # get prediction for the specified class

        
    def __call__(self, x, **kwargs):
        '''
        args:
            x: superpixel indicator: numpy array
            w: baseline value to set for "null" pixels.
        '''
        if self.baseline is None: raise Exception('Need to first initialize baseline in evaluation function!')
        
        if self.shapley_baseline == 'mean':
            ## Baseline
            w = torch.zeros((x.shape[0], self.c, self.h, self.w))
            self.data_iterator = iter(self.dataloader)
            for i in range(x.shape[0]):
                try:
                    data, target = next(self.data_iterator)
                except StopIteration:
                    self.data_iterator = iter(self.dataloader)
                    data, target = next(self.data_iterator)
                w[i, ...] = data[0, ...]
            w = tensor2cuda(w)
        
        # Interaction Shapley---------------------------------
        if self.j is not None:                               #
            if self.fixed_present:                           #
                j_vector = np.ones((x.shape[0], 1))    
                x = np_insert(x, j_vector, index = self.j)   #
            else:                                            #
                j_vector = np.zeros((x.shape[0], 1))         #
                x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------
        
        with torch.no_grad():

            x = numpy2cuda(x)
            mask, _ = self.sp_mapping(x, x_orig = self.x_baseline, segment_mask = self.segment_mask)
            mask = tensor2cuda(mask)

            x = torch.mul(mask, self.x_baseline) +  (1-mask) * self.baseline_scaling
            if self.shapley_baseline == 'mean': x += torch.mul(1-mask, w)
            # ########################33
            # print('=============================')
            # import matplotlib.pyplot as plt
            # # set figure size
            # plt.rcParams['figure.figsize'] = [3, 3]
            # image = tensor2numpy(x)[0,...].transpose((1, 2, 0))
            # plt.imshow(image, cmap = 'gray')
            # plt.show()
            # ########################33




            pred = self.model(x)
            if self.binary:
                # outdated. we want output to be vector.
                pred = torch.sigmoid(pred)
                if self.baseline < 0.5: pred = 1-pred
            else:
                # pred = torch.exp(-F.cross_entropy(pred, self.baseline.expand(pred.shape[0]), reduction = 'none')) # get probability of the predicted class of the original sample
                pred= pred[:,self.baseline].reshape(-1)
            pred = pred.cpu().detach().numpy()

        # return np.vstack((1-pred, pred)).transpose()
        return np.vstack((np.zeros_like(pred), pred)).transpose()

def superpixel_to_mask(x_superpixel, x_orig, segment_mask = None, verbose = False):
    '''
    input:
        x_superpixel: binary matrix, n x num_superpixels
        x_orig: image tensor, 1 x c x h x w
        segment_mask: optional to save redundant calculations, Represents a flattened binary mask for each superpixel group

    '''

    x_superpixel = tensor2cuda(x_superpixel).float()

    #Note: SLIC requires image to be in h x w x c
    _, c, h, w = x_orig.shape
    #image = x_orig[0, ...].permute(1, 2, 0).cpu()  # h x w x c
    #mask_out = torch.zeros((x_superpixel.shape[0], h, w, c)) # n x h x w x c
    num_segments = x_superpixel.shape[1]

    if segment_mask is None:

        image = x_orig[0, ...].permute(1, 2, 0).cpu()  # h x w x c
        segments = slic(image, n_segments = num_segments, sigma = 5, start_label = 0)
        if verbose: print('# Superpixels Actual: %s' % str(segments.max()+1))
        segment_mask = []
        for i in range(num_segments):
            segment_mask.append(torch.tensor(segments == i, dtype = torch.float32).unsqueeze(0)) # 1 x h x w
        segment_mask = tensor2cuda(torch.cat(segment_mask, dim = 0)) #  num_superpixels x h x w
        segment_mask = segment_mask.unsqueeze(0).expand(x_superpixel.shape[0], -1, -1, -1) # n x num_superpixels x h x w
        segment_mask = torch.flatten(segment_mask, 2)  # n x num_superpixels x (h*w)

    mask_out = torch.matmul(x_superpixel.unsqueeze(1), segment_mask).squeeze(1)  # n x (h*w)
    mask_out = mask_out.reshape(x_superpixel.shape[0], h, w)  # n x h x w
    mask_out = mask_out.unsqueeze(1).expand(-1, c, -1, -1) # n x c x h x w   copy over channels
    ''' 
    for j in range(x_superpixel.shape[0]):
        for i in range(num_segments):
            if x_superpixel[j, i] == 1:
                segment_mask = torch.tensor(segments == i, dtype = torch.int32).unsqueeze(-1).expand(-1, -1, c)
                mask_out[j, ...] = mask_out[j, ...] + segment_mask
    return mask_out.permute(0, 3, 1, 2) # n x c x h x w
    '''
    return mask_out, segment_mask 

def sample_mnist(index = 0, baseline = 'zero', num_superpixels = 0, data_path = './Files/Data', **kwargs):
    '''
    draws a sample from the mnist binary dataset, and calculates shapley values

    args:
        index: index value of sample drawn from MNIST test set

    ''' 
    # parameters
    model_path = './MNIST/MLP_baseline.pt'
    if num_superpixels >0:
        sp_mapping = superpixel_to_mask
        node_labels = np.arange(num_superpixels).tolist()
    else:
        sp_mapping = None
        node_labels = np.arange(28*28).tolist()

    # Data Sample
    dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 0)
    dataiterator = iter(dataloader)
    for i in range(index+1):
        x, label = next(dataiterator)


    mask_out, segment_mask = sp_mapping(torch.ones((1,num_superpixels)), x)
    x = x.cpu().detach().numpy()
    print('Label: %s' % str(label))

    ############################
    # print sample and prediction

    print('=============================')
    import matplotlib.pyplot as plt
    # set figure size
    plt.rcParams['figure.figsize'] = [3, 3]
    image = x[0,...].transpose((1, 2, 0))
    plt.imshow(image, cmap = 'gray')
    plt.show()

    ###########################
    # Initialize Explainer
    # Explainer = Image_Explainer(binary_pred = False, model_path = model_path, baseline = baseline, num_superpixels= num_superpixels, sp_mapping = sp_mapping, dataset = dataset, **kwargs)
    # shapley_values, shapley_matrix_pairwise = Explainer(x)


    ###########################
    # Print
    # print('Prediction Probability: ' + str(Explainer.value_function.baseline.item()))

    # print('unary shapley:')
    # print(np.round(shapley_values, 3))

    # print('pairwise:')
    # print(np.round(shapley_matrix_pairwise, 3))

    # db = {
    # 'node_labels': node_labels,
    # 'x': x, 
    # 'label': label
    # }

    return [node_labels, x, label, mask_out, segment_mask]

def run_bayesshap(x_test, x_train, y_test, f_blackbox, method = 'mnist', explainer = 'bayesshap'):
    n_features = x_test.shape[1]
    if method == 'mnist' or method == 'fmnist':
        feature_selection = False # to reduce computation time
        bayesshap_l2 = False
        datatype = 'image'
    else:
        feature_selection = False
        bayesshap_l2 = False
        datatype = 'tabular'


    if explainer == 'bayesshap':
        kernel = 'shap'
    else:
        kernel = 'lime'

    sys.path.append('../Modeling-Uncertainty-Local-Explainability')
    from bayes.explanations import BayesLocalExplanations, explain_many
    from bayes.data_routines import get_dataset_by_name
    exp_init = BayesLocalExplanations(training_data=x_train,
                                                data=datatype,
                                                kernel=kernel,
                                                categorical_features=np.arange(x_train.shape[1]),
                                                verbose=True)
    interval_list = []
    attr_list = []
    for i,x_sample in tqdm(enumerate(x_test), total = x_test.shape[0]):
        # if only calculating for certain samples
        rout = exp_init.explain(classifier_f=f_blackbox,
                                data=x_test[i,:],
                                # y_test is a n-dimensional vector
                                label=int(y_test[i]),
                                #cred_width=cred_width,
                                n_samples = 200,
                                focus_sample=False,
                                feature_selection = feature_selection,
                                n_features = n_features,
                                enumerate_initial = False,
                                l2=bayesshap_l2)

        interval_list.append(rout['blr'].creds)
        attr_list.append(rout['blr'].coef_)
    unc_list = np.array(interval_list)
    attr_list = np.array(attr_list)

    return attr_list, unc_list