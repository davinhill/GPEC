import numpy as np
import pandas as pd
#from sympy import Q
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
import time
import warnings

sns.set_theme()
rc={'font.size': 19, 'axes.labelsize': 20, 'legend.fontsize': 18, 
    'axes.titlesize': 21, 'xtick.labelsize': 17, 'ytick.labelsize': 17}
sns.set(rc=rc)
sns.set_style('white')

from tqdm import tqdm
from datetime import datetime
import os
import sys
from argparse import ArgumentParser 

os.chdir('../../') # change to root directory of the project
sys.path.append('./')
#from Models import * # test models
from GPEC.utils import * # utility functions
from GPEC import * # GPEC functions
from GPEC.utils import utils_tests # utility functions



parser = ArgumentParser(description='Kernel Tests')

parser.add_argument('--method', type = str,default='cosinv',
                    help='germancredit_3_1')

parser.add_argument('--explainer', type = str,default='bayesshap',
                    help='')

parser.add_argument('--n_train_samples', type = int,default=100, help='number of training samples for GP')
parser.add_argument('--lam', type = float,default=0.5,
                    help='lambda parameter for kernel')
parser.add_argument('--rho', type = float,default=0.5,
                    help='rho parameter for kernel')
parser.add_argument('--n_test_samples', type=int, default=10000, help='number of test samples')
parser.add_argument('--n_iterations', type = int, default = 50)
parser.add_argument('--kernel', type = str,default='RBF',
                    help='')
parser.add_argument('--kernel_normalization', type = int,default=1, help='normalize kernel s.t. k(x,x)=1')
parser.add_argument('--max_batch_size', type = int,default=1024, help='Max number of GPs to train simultaneously. Number of batches == #features / max_batch_size')
parser.add_argument('--plot_explanations', type = int,default=1, help='flag to plot explanations')
parser.add_argument('--plot_flag', type = int,default=0, help='flag to save plots. overrides plot_explanations.')
parser.add_argument('--plot_feat', type = int,default=0, help='which feature to plot (0 or 1)')
parser.add_argument('--save_data', type = int,default=1, help='flag to save output')

#########
parser.add_argument('--use_gpec', type = int,default=0, help='flag to use GPEC')
parser.add_argument('--use_labelnoise', type = int,default=0, help='flag to use label noise (if using GPEC). only implemented for bayesshap, bayeslime, cxplain.')
parser.add_argument('--n_labelnoise_samples', type = int,default=10, help='if using labelnoise and explainer does not return uncertainty. Number of explanations to get from explainer for uncertainty estimate.')
parser.add_argument('--n_mc_samples', type = int,default=200, help='number of samples for approximating explanations')
parser.add_argument('--gpec_lr', type = float,default=1.0,help='Learning Rate for GPEC')
parser.add_argument('--learn_noise', type = int,default= 0, help='learn additional heteroskedastic GP noise for labels')
parser.add_argument('--adhoc_str', type = str,default= '', help='additional string for saving ad-hoc tests')

args = parser.parse_args()
utils_io.print_args(args)

# cxplain, bayeslime, bayesshap can use labelnoise. and they can export uncertainty.
# shapleysampling can use labelnoise. it cannot export uncertainty.
# kernelshap is not implemented. it cannot export uncertainty.
if args.use_labelnoise == 1 and args.explainer == 'kernelshap':
    raise ValueError('LabelNoise not implemented for KernelSHAP.')
if args.use_gpec == 0 and args.explainer == 'kernelshap':
    warnings.warn('KernelSHAP does not have uncertainty estimate by itself.')

if args.use_labelnoise == 1:
    n_labelnoise_samples = args.n_labelnoise_samples
else:
    n_labelnoise_samples = 1

lam = args.lam
rho = args.rho
if args.use_gpec == 0:
    lam = rho = 'NA'
plotfeat = args.plot_feat

if args.kernel_normalization == 1:
    kernel_normalization = True
else:
    kernel_normalization = False

plot_train = True
if args.explainer == 'kernelshap':
    output_shape = 'singleclass'
elif args.explainer == 'lime':
    output_shape = 'multiclass'
else:
    output_shape = 'multiclass'

'''
###############################################
  _____        _           _____      _               
 |  __ \      | |         / ____|    | |              
 | |  | | __ _| |_ __ _  | (___   ___| |_ _   _ _ __  
 | |  | |/ _` | __/ _` |  \___ \ / _ \ __| | | | '_ \ 
 | |__| | (_| | || (_| |  ____) |  __/ |_| |_| | |_) |
 |_____/ \__,_|\__\__,_| |_____/ \___|\__|\__,_| .__/ 
                                               | |    
                                               |_|    
###############################################
'''

if args.method == 'cosinv':

    from Tests.Models import synthetic_cosinv
    f_blackbox = synthetic_cosinv.model(output_shape = output_shape, sigmoid = True)

    dataset_name = 'cosinv'
    post_str = ''
    geo_matrix = np.load('./Files/Models/%s_geomatrix%s.npy' % (dataset_name, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples%s.npy' % (dataset_name, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')
    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

    # synthetic test data
    xmin, xmax, ymin, ymax = x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max()
    xmax = ymax = 10
    xmin = -10
    ymin = -10
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1

    feat1 = 'x1'
    feat2 = 'x2'
    decision_threshold = 0
    xmin, xmax, ymin, ymax = x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max()
    axislim = [xmin, xmax, ymin, ymax]

if args.method == 'abs':

    from Tests.Models import synthetic_abs
    f_blackbox = synthetic_abs.model(output_shape = output_shape)

    dataset_name = 'abs'
    post_str = ''
    geo_matrix = np.load('./Files/Models/%s_geomatrix%s.npy' % (dataset_name, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples%s.npy' % (dataset_name, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')
    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

    # synthetic test data
    xmin, xmax, ymin, ymax = x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max()
    xmax = ymax = 10
    xmin = ymin = -10
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1

    feat1 = 'x1'
    feat2 = 'x2'
    decision_threshold = 0
    axislim = [xmin, xmax, ymin, ymax]
if args.method == 'linear':

    from Tests.Models import synthetic_linear
    f_blackbox = synthetic_linear.model(output_shape = output_shape)

    dataset_name = 'linear'
    post_str = ''
    geo_matrix = np.load('./Files/Models/%s_geomatrix%s.npy' % (dataset_name, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples%s.npy' % (dataset_name, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')
    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

    # synthetic test data
    xmin, xmax, ymin, ymax = x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max()
    xmax = ymax = 10
    xmin = ymin = -10
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1

    feat1 = 'x1'
    feat2 = 'x2'
    decision_threshold = 0
    axislim = [xmin, xmax, ymin, ymax]
elif args.method[:6] == 'census':

    if args.method == 'census_Age_Hours':
        feat1 = 'Age'
        feat2 = 'Hours per week'
        post_str = ''
        axislim = [20, 70, 20, 75] # axislim = [xmin, xmax, ymin, ymax]
    elif args.method == 'census_Age_Education':
        feat1 = 'Age'
        feat2 = 'Education-Num'
        post_str = ''
        axislim = [20, 70, 8, 16] # axislim = [xmin, xmax, ymin, ymax]
    elif args.method == 'census_Age_Hours_reg':
        feat1 = 'Age'
        feat2 = 'Hours per week'
        post_str = '_reg'
        axislim = [20, 70, 20, 75] # axislim = [xmin, xmax, ymin, ymax]
    dataset_name = 'census'

    # Load Pretrained Model
    from Tests.Models import xgb_models
    model_path = './Files/Models/model_census_%s_%s%s.json' % (feat1, feat2, post_str)
    f_blackbox = xgb_models.xgboost_wrapper(model_path, output_shape = output_shape)

    # Load Geo Matrix and Manifold Samples
    geo_matrix = np.load('./Files/Models/%s_geomatrix_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))

    # Load Data
    x_train = pd.read_pickle('./Files/Data/%s_x_train.pkl' % dataset_name)
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv'% dataset_name)
    x_test = pd.read_pickle('./Files/Data/%s_x_test.pkl'% dataset_name)
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv'% dataset_name)
    x_train = x_train[[feat1, feat2]].to_numpy()
    x_test = x_test[[feat1, feat2]].to_numpy()

    # Create synthetic test data
    xmin, xmax, ymin, ymax = x_train[:,0].min()*1.2, x_train[:,0].max()*0.8, x_train[:,1].min(), x_train[:,1].max()
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1

    decision_threshold = 0.5

elif args.method[:6] == 'german':

    dataset_name = 'germancredit'
    if args.method == '%s_3_1' % dataset_name:
        feat1 = 3
        feat2 = 1
        post_str = ''
        axislim = [0, 150, 0, 50] # axislim = [xmin, xmax, ymin, ymax]

    from Tests.Models import xgb_models
    model_path = './Files/Models/model_%s_%s_%s%s.json' % (dataset_name, feat1, feat2, post_str)
    f_blackbox = xgb_models.xgboost_wrapper(model_path, output_shape = output_shape)

    geo_matrix = np.load('./Files/Models/%s_geomatrix_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')
    x_train = x_train[:,[feat1, feat2]]
    x_test = x_test[:,[feat1, feat2]]

    # synthetic test data
    xmin, xmax = 0,170
    ymin, ymax = 0,60
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1


    decision_threshold = 0.5

elif args.method[:6] == 'nhanes':

    dataset_name = 'nhanes'
    if args.method == '%s_Age_Blood' % dataset_name:
        feat1 = 'Age'
        feat2 = 'Blood Urea Nitrogen'
        post_str = ''
        axislim = [20, 80, 0, 18] # axislim = [xmin, xmax, ymin, ymax]

    from Tests.Models import xgb_models
    model_path = './Files/Models/model_%s_%s_%s%s.json' % (dataset_name, feat1, feat2, post_str)
    f_blackbox = xgb_models.xgboost_wrapper(model_path, output_shape = output_shape)

    geo_matrix = np.load('./Files/Models/%s_geomatrix_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')
    colnames = pd.read_pickle('./Files/Data/%s_colnames.pkl' % (dataset_name))
    x_train = x_train[:,[colnames.index(feat1),colnames.index(feat2)]]
    x_test = x_test[:,[colnames.index(feat1),colnames.index(feat2)]]

    # synthetic test data
    xmin, xmax =  20,80
    ymin, ymax = 0,18
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1


    decision_threshold = 0.5
elif args.method[:6] == 'online':

    dataset_name = 'onlineshoppers'
    if args.method == '%s_4_8' % dataset_name:
        feat1 = 4
        feat2 = 8
        post_str = ''
        axislim = [0, 100, 0, 80] # axislim = [xmin, xmax, ymin, ymax]

    from Tests.Models import xgb_models
    model_path = './Files/Models/model_%s_%s_%s%s.json' % (dataset_name, feat1, feat2, post_str)
    f_blackbox = xgb_models.xgboost_wrapper(model_path, output_shape = output_shape)

    geo_matrix = np.load('./Files/Models/%s_geomatrix_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')
    x_train = x_train[:,[feat1, feat2]]
    x_test = x_test[:,[feat1, feat2]]

    # synthetic test data
    xmin, xmax = 0,150
    ymin, ymax = 0, 80
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1

    decision_threshold = 0.5


# limit train/test samples if specified
x_train, y_train = utils_np.subsample_rows(matrix1 = x_train, matrix2 = y_train, max_rows = args.n_train_samples)
x_test,y_test = utils_np.subsample_rows(matrix1 = x_test, matrix2 = y_test, max_rows = args.n_test_samples)



gpec = GPEC.GPEC_Explainer(
    f_blackbox,
    x_train,
    y_train,
    explain_method = args.explainer,
    use_gpec = (args.use_gpec == 1),
    kernel = args.kernel,
    lam = args.lam,
    rho = args.rho,
    kernel_normalization = kernel_normalization,
    max_batch_size = 1024,
    gpec_lr = args.gpec_lr,
    gpec_iterations = args.n_iterations,
    use_labelnoise = (args.use_labelnoise == 1),
    learn_addn_noise = (args.learn_noise == 1),
    n_mc_samples = 200,
    scale_data = False,
    calc_attr_during_pred = (args.use_gpec == 0), # don't calculate explanations when using gpec (not used in figure)

    manifold_samples = manifold_samples, # optional: precomputed boundary samples
    geo_matrix = geo_matrix, # optional: precomputed geodesic matrix

    )


attr_list, var_list, ci_list = gpec.explain(x_test, y_test)

'''
###############################################
   _____                 
  / ____|                
 | (___   __ ___   _____ 
  \___ \ / _` \ \ / / _ \
  ____) | (_| |\ V /  __/
 |_____/ \__,_| \_/ \___|
###############################################
'''
feat_list = [plotfeat]
if args.plot_flag == 1:
    ###############################################
    # Plot
    ###############################################
    #sns.cubehelix_palette(as_cmap=True)
    # coolwarm

    plot_unc_list = ci_list
    filename = '_'.join([
    args.kernel,
    args.method,
    args.explainer,
    'rho'+str(rho),
    'lam'+str(lam),
    ])
    save_path = './Files/Results/uncertaintyplot/%s/%s/%s/%s/%s.jpg' % (args.method, 'plotfeat'+ str(plotfeat), 'gpec'+str(args.use_gpec), 'labelnoise'+str(args.use_labelnoise), filename)
    utils_tests.uncertaintyplot(x_train = x_train, x_test = x_test, hue_list = plot_unc_list, save_path = save_path, f_blackbox = f_blackbox, feat_list = feat_list, rho = args.rho, lam = args.lam, plot_train = True, axislim = axislim)

    ###############################################
    # Plot Explanations
    ###############################################
    if args.plot_explanations == 1:
        if args.explainer not in ['kernelshap', 'lime']:
            raise ValueError('plot_explanations not yet implemented')
        
        # plot uncertainty
        # exp_test = attr_list # get test explanations
        exp_test = attr_list[:,feat_list] # plot only one feature

        save_path = './Files/Results/uncertaintyplot/%s/%s/%s/explanations_%s.jpg' % (args.method, 'plotfeat'+ str(plotfeat) ,str(args.kernel_normalization), filename)
        utils_tests.uncertaintyplot(x_train = x_train, x_test = x_test, hue_list = exp_test, save_path = save_path, f_blackbox = f_blackbox, feat_list = feat_list, cmap = cm.coolwarm, rho = args.rho, lam = args.lam, plot_train = True, center_cmap = True, center = 0)
        
    # Plot model output
    if output_shape == 'multiclass':
        output_list = f_blackbox(x_test)[:,1].reshape(-1,1)
    else:
        output_list = f_blackbox(x_test).reshape(-1,1)

    save_path = './Files/Results/uncertaintyplot/%s/%s/%s/%s/output_%s.jpg' % (args.method, 'plotfeat'+ str(plotfeat), str(args.use_gpec), str(args.use_labelnoise), str(args.method))
    utils_tests.uncertaintyplot(x_train = x_train, x_test = x_test, hue_list = output_list, save_path = save_path, f_blackbox = f_blackbox, feat_list = feat_list, cmap = cm.coolwarm, rho = args.rho, lam = args.lam, plot_train = True, center_cmap=True, center = decision_threshold, axislim = axislim)


###############################################
# Save Data for Figure
###############################################
if args.save_data == 1:
    '''
    if args.plot_flag ==0:
        exp_test = explainer(x_test)
        exp_test = exp_test[:,feat_list]
    '''
    # Plot model output
    if output_shape == 'multiclass':
        output_list = f_blackbox(x_test)[:,1].reshape(-1,1)
    else:
        output_list = f_blackbox(x_test).reshape(-1,1)

    saved_data = {
        'args': args,
        'x_train': x_train,
        'x_test': x_test,
        #'gpec_ci_list': gpec_ci_list, # estimated uncertainty for each explanation
        #'gpec_var_list': gpec_var_list, # estimated uncertainty for each explanation
        #'gpec_attr_list': gpec_attr_list, # predicted explanations from GPEX
        'attr_list': attr_list, # explanations from explainer
        'ci_list': ci_list, # ci from explainer
        'var_list': var_list, # variance from explainer
        'output_list': output_list, # black-box model output for test points
        #'GT_list': exp_test, # ground truth explanations
        'rho': rho,
        'lam': lam,
        'method': args.method,
        'explainer': args.explainer,
        'kernel': args.kernel,
        'xx': xx,
        'yy': yy,
        'feat1': feat1,
        'feat2': feat2,
        #'time_train': time_train,
        #'time_pred': time_pred,
        #'time_attr': time_mean,
        #'time_var': time_var,
        #'time_ci': time_ci,
    }

    filename = '_'.join([
        args.method,
        args.explainer,
        args.kernel,
        'rho'+str(rho),
        'lam'+str(lam),
        'uselabelnoise' + str(args.use_labelnoise),
        'mcsamples' + str(args.n_mc_samples),
        ])
    prepend = ''
    if args.use_labelnoise: prepend = 'labelnoise'
    preped = prepend + args.adhoc_str
    save_path = './Files/Results/uncertaintyplot/saved_results_%s/%s.pkl' % (prepend, filename)
    foldername = os.path.dirname(save_path)
    utils_io.make_dir(foldername)
    utils_io.save_dict(saved_data, save_path)
    print(save_path)
    print('done!')