import numpy as np
import warnings
try:
    import torch
except:
    warnings.warn('pytorch not installed.')


import time
try:
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
except:
    warnings.warn('sklearn not installed.')

try:
    import xgboost
except:
    warnings.warn('sklearn not installed.')

from tqdm import tqdm
from datetime import datetime
import os
import sys
from argparse import ArgumentParser 

os.chdir('../../') # change to root directory of the project
sys.path.append('./')
try:
    from GPEC import * # GPEC functions
except:
    warnings.warn('Errors Loading GPEC')

try:
    from GPEC.utils import * # utility functions
    from GPEC import decision_boundary
except:
    warnings.warn('Errors Loading GPEC utilities')

from GPEC.utils import utils_torch
from GPEC.utils import utils_io
from GPEC.utils import utils_np
from GPEC import explainers

parser = ArgumentParser(description='Kernel Tests')

parser.add_argument('--method', type = str,default='cifar10',
                    help='census_Age_Hours')

parser.add_argument('--explainer', type = str,default='kernelshap',
                    help='')

parser.add_argument('--n_train_samples', type = int,default=100,
                    help='number of training samples for GP')
parser.add_argument('--lam', type = float,default=1.0,
                    help='lambda parameter for kernel')
parser.add_argument('--rho', type = float,default=0.001,
                    help='rho parameter for kernel')
parser.add_argument('--n_test_samples', type=int, default=100,
                    help='number of test samples')
parser.add_argument('--n_iterations', type = int, default = 100)
parser.add_argument('--kernel', type = str,default='WEG',
                    help='')
parser.add_argument('--kernel_normalization', type = int,default=1, help='normalize kernel s.t. k(x,x)=1')
parser.add_argument('--max_batch_size', type = int,default=1024, help='Max number of GPs to train simultaneously. Number of batches == #features / max_batch_size')
parser.add_argument('--sample_batch_size', type = int,default=100, help='GPEC batch size of samples during test')
parser.add_argument('--seed', type = int,default=0, help='kfold seed')
parser.add_argument('--gamma', type = float,default=0.0,help='xgboost gamma parameter')
parser.add_argument('--l2_reg', type = float,default=0.0,help='nn l2 regularization parameter')
parser.add_argument('--nn_epochs', type = int,default=30,help='number of epochs for nn training')
parser.add_argument('--nn_retrain', type = int,default=0,help='for neural networks. 0: check if saved model exists. 1: always retrain NN')
parser.add_argument('--bayesshap_idx_start', type = int,default=-1,help='start idx for calculating bayesshap / bayeslime. Setting this to -1 disables it.')
parser.add_argument('--gpec_lr', type = float,default=1.0,help='Learning Rate for GPEC')
parser.add_argument('--learn_noise', type = int,default=0, help='learn GP noise')
parser.add_argument('--nn_softplus_beta', type = float,default=0.0, help='softplus beta parameter for nn training')

args = parser.parse_args()
utils_io.print_args(args)

lam = args.lam
rho = args.rho
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
time_manifold = 0

if args.kernel_normalization == 1:
    kernel_normalization = True
else:
    kernel_normalization = False

if args.explainer == 'kernelshap' and args.method not in ['mnist', 'fmnist', 'cifar10']:
    output_shape = 'singleclass'
else:
    output_shape = 'multiclass'

gpec_explainer_list = ['kernelshap', 'lime', 'deepshap']
geo_matrix = None
manifold_samples = None


###############################################
# Define Functions
###############################################
if args.method[:6] == 'census':

    dataset_name = 'census'
    post_str = ''
    n_classes = 2

    #####################
    # Load Data
    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)

    #####################
    # Train XGBoost Model
    d_train = xgboost.DMatrix(x_train, label=y_train)
    d_test = xgboost.DMatrix(x_test, label=y_test)

    params = {
        "eta": 0.5,
        "objective": "binary:logistic",
        # "subsample": 0.5,
        "gamma": args.gamma,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss"
    }

    model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

    pred =(model.predict(d_train) >= 0.5)*1
    train_accy = metrics.accuracy_score(y_train, pred)
    print('train accy: %s' % str(metrics.accuracy_score(y_train, pred)))

    pred =(model.predict(d_test) >= 0.5)*1
    test_accy = metrics.accuracy_score(y_test, pred)
    print('test accy: %s' % str(metrics.accuracy_score(y_test, pred)))
    print('done!')

    from Tests.Models import xgb_models
    f_blackbox = xgb_models.xgboost_wrapper(model = model, output_shape = output_shape)

    #####################
    # Calculate manifold samples and geo matrix
    time_start = time.time()
    if args.kernel == 'WEG' and args.explainer in gpec_explainer_list:
        # _, seed_samples = decision_boundary.create_grid(x_train, gridsize = 2)
        seed_samples = decision_boundary.advsamples_xgboost(x_train, y_train, xgb_model = model,n_samples = 250)
        manifold_samples = decision_boundary.sampledb_DBPS_binary(seed_samples, f_blackbox, decision_threshold = 0.5, n_samples_per_class = 100, batch_size = 4096)
        geo_matrix = decision_boundary.geo_kernel_matrix(manifold_samples)
    time_manifold = time.time() - time_start

    #####################

elif args.method[:6] == 'german':

    dataset_name = 'germancredit'
    post_str = ''
    n_classes = 2
    #####################
    # Load Data
    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)
    #####################
    # Train XGBoost Model
    d_train = xgboost.DMatrix(x_train, label=y_train)
    d_test = xgboost.DMatrix(x_test, label=y_test)

    params = {
        "eta": 0.5,
        "objective": "binary:logistic",
        # "subsample": 0.5,
        "gamma": args.gamma,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss"
    }

    model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

    pred =(model.predict(d_train) >= 0.5)*1
    train_accy = metrics.accuracy_score(y_train, pred)
    print('train accy: %s' % str(metrics.accuracy_score(y_train, pred)))

    pred =(model.predict(d_test) >= 0.5)*1
    test_accy = metrics.accuracy_score(y_test, pred)
    print('test accy: %s' % str(metrics.accuracy_score(y_test, pred)))
    print('done!')

    from Tests.Models import xgb_models
    f_blackbox = xgb_models.xgboost_wrapper(model = model, output_shape = output_shape)

    time_start = time.time()
    ########################################
    # Calculate Manifold Samples

    if args.kernel == 'WEG' and args.explainer in gpec_explainer_list:
        # assume data is n x d
        # xx_list, grid = decision_boundary.create_grid(x_train, gridsize = 2)
        seed_samples = decision_boundary.advsamples_xgboost(x_train, y_train, xgb_model = model,n_samples = 250)
        manifold_samples = decision_boundary.sampledb_DBPS_binary(seed_samples, f_blackbox, decision_threshold = 0.5, n_samples_per_class = 100, batch_size = 4096)
        geo_matrix = decision_boundary.geo_kernel_matrix(manifold_samples)

    #####################
    time_manifold = time.time() - time_start

elif args.method[:6] == 'online':

    dataset_name = 'onlineshoppers'
    post_str = ''
    n_classes = 2

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)
    #####################
    # Train XGBoost Model
    d_train = xgboost.DMatrix(x_train, label=y_train)
    d_test = xgboost.DMatrix(x_test, label=y_test)

    params = {
        "eta": 0.5,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "gamma": args.gamma,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss"
    }

    model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

    pred =(model.predict(d_train) >= 0.5)*1
    train_accy = metrics.accuracy_score(y_train, pred)
    print('train accy: %s' % str(metrics.accuracy_score(y_train, pred)))

    pred =(model.predict(d_test) >= 0.5)*1
    test_accy = metrics.accuracy_score(y_test, pred)
    print('test accy: %s' % str(metrics.accuracy_score(y_test, pred)))
    print('done!')

    from Tests.Models import xgb_models
    f_blackbox = xgb_models.xgboost_wrapper(model = model, output_shape = output_shape)

    time_start = time.time()

    ########################################
    # Calculate Manifold Samples
    # assume data is n x d
    if args.kernel == 'WEG' and args.explainer in gpec_explainer_list:
        # xx_list, seed_samples = decision_boundary.create_grid(x_train, gridsize = 2)
        seed_samples = decision_boundary.advsamples_xgboost(x_train, y_train, xgb_model = model,n_samples = 250)
        manifold_samples = decision_boundary.sampledb_DBPS_binary(seed_samples, f_blackbox, decision_threshold = 0.5, n_samples_per_class = 100, batch_size = 4096)
        geo_matrix = decision_boundary.geo_kernel_matrix(manifold_samples)

    #####################
    time_manifold = time.time() - time_start

elif args.method[:6] == 'nhanes':

    dataset_name = 'nhanes'
    post_str = ''
    n_classes = 2

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)
    #####################
    # Train XGBoost Model
    d_train = xgboost.DMatrix(x_train, label=y_train)
    d_test = xgboost.DMatrix(x_test, label=y_test)

    params = {
        "eta": 0.1,
        "objective": "binary:logistic",
        "subsample": 1.0,
        "gamma": args.gamma,
        # "lambda": 10,
        "max_depth": 4,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss",
        "min_child_weight": 5,
        "colsample_bytree": 1.0,
        # "max_depth": 6
    }

    model = xgboost.train(params, d_train, 10000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

    pred =(model.predict(d_train) >= 0.5)*1
    train_accy = metrics.accuracy_score(y_train, pred)
    print('train accy: %s' % str(metrics.accuracy_score(y_train, pred)))

    pred =(model.predict(d_test) >= 0.5)*1
    test_accy = metrics.accuracy_score(y_test, pred)
    print('test accy: %s' % str(metrics.accuracy_score(y_test, pred)))
    print('done!')

    from Tests.Models import xgb_models
    f_blackbox = xgb_models.xgboost_wrapper(model = model, output_shape = output_shape)

    time_start = time.time()

    ########################################
    # Calculate Manifold Samples
    # assume data is n x d
    if args.kernel == 'WEG' and args.explainer in gpec_explainer_list:
        xx_list, seed_samples = decision_boundary.create_grid(x_train, gridsize = 2)
        # seed_samples = decision_boundary.advsamples_xgboost(x_train, y_train, xgb_model = model,n_samples = 250)
        manifold_samples = decision_boundary.sampledb_DBPS_binary(seed_samples, f_blackbox, decision_threshold = 0.5, n_samples_per_class = 100, batch_size = 4096)
        geo_matrix = decision_boundary.geo_kernel_matrix(manifold_samples)

    #####################
    time_manifold = time.time() - time_start



elif args.method[:5] == 'mnist':
    sys.path.append('/Tests/Models/blackbox_model_training/image/')
    from Tests.Models.blackbox_model_training.image.train import train_network
    from nn_datasets import load_mnist
    from model import SimpleANN

    dataset_name = 'mnist'
    post_str = ''
    n_classes = 10
    nn_params = {
        # Adversarial Attack Params
        'num_attack_untarg': 500, # number of examples coming from undirected tack on own class trian pts
        'num_attack_targ': 50, # number of examples coming from directed from other class
        'bounds': (0,1), # min/max value bounds for your inputs. default is (0,1) for images. converted to tuple later.
        'attack_norm_untarg': 'LinfPGD',
        'attack_norm_targ': 'LinfPGD',
        'batch_size_adv': 256, # number of adv ex to compute in parallel at a time
        'epsilons': [0.0,0.0002, 0.0005,0.0008, 0.001,0.0015,0.002,0.003,0.01,0.1,0.3,0.5,1.0,], # list of ball sizes in which to search for adv exs
        'eps_use': 'first', # options: first

        # NN Training Params
        'network': 'SimpleANN', 
        'network_dimensions': '700-400-200', 
        'batch_size': 64, 
        'input_dim': 784, 
        'feature_dim': 100, 
        'num_classes': 10, 
        'epochs': 30, 
        'lr': 2, 
        'optimizer': 'SGD', 
        'milestones': [8,15], # lr scaling
        'lr_gamma': 0.5, # lr scaling at each milestone
        'l2_reg': args.l2_reg, # l2 regularization
        'nn_softplus_beta': args.nn_softplus_beta, # softplus beta paramter
    }
    nn_params = utils_io.dict_to_argparse(nn_params)
    nn_params.l2_reg = float(nn_params.l2_reg)
    nn_params.nn_softplus_beta = float(nn_params.nn_softplus_beta)
    #nn_params.l2_reg = np.exp(nn_params.l2_reg)

    # Load Data
    tr_data_im, tr_data_lab, _, tr_loader, _, _, _, _, te_data_im, te_data_lab, _, te_loader = load_mnist(nn_params)

    #####################
    # Train nn

    curdir = './Files/Models/Images/MNIST-Trained-Model_%s_%s' % (str(nn_params.l2_reg), str(nn_params.nn_softplus_beta))
    utils_io.make_dir(curdir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: %s' % device)
    model = SimpleANN(**nn_params.__dict__).to(device)
    
    # train model
    if args.nn_retrain == 1:
        model, train_accy, test_accy = train_network(model, nn_params.epochs, tr_loader, te_loader, nn_params.lr, nn_params.optimizer, nn_params.milestones, nn_params.lr_gamma, curdir, nn_params.l2_reg)
        print('train accy: %s' % str(train_accy))
        print('test accy: %s' % str(test_accy))
    else:
        try:
            model.load_state_dict(torch.load(curdir + '/model.pth', map_location = torch.device(device)))
            print('Model Loaded!')
        except:
            warnings.warn('Error in loading saved model. Retraining...')
            model, train_accy, test_accy = train_network(model, nn_params.epochs, tr_loader, te_loader, nn_params.lr, nn_params.optimizer, nn_params.milestones, nn_params.lr_gamma, curdir, nn_params.l2_reg)
            print('train accy: %s' % str(train_accy))
            print('test accy: %s' % str(test_accy))

    # evaluate
    model.eval()
    print('done!')

    from Tests.Models import nn_wrapper
    f_blackbox = nn_wrapper.nn_wrapper(model = model, output_shape = output_shape)
    time_start = time.time()
    ########################################
    # Calculate Manifold Samples
    ########################################

    # assume data is n x d
    if args.kernel == 'WEG' and args.explainer in gpec_explainer_list:
        if args.nn_retrain == 1:
            tmp = decision_boundary.gpec_db_nn(model, tr_data_im, tr_data_lab, tr_loader, nn_params, save = True, save_path = curdir + '/db.pkl')
        else:
            try:
                tmp = utils_io.load_dict(curdir + '/db.pkl')
            except:
                tmp = decision_boundary.gpec_db_nn(model, tr_data_im, tr_data_lab, tr_loader, nn_params, save = True, save_path = curdir + '/db.pkl')
        manifold_samples_dict, geo_matrix_dict, time_manifold = tmp['manifold_samples'], tmp['geo_matrix'], tmp['time']
    
    x_train = utils_torch.tensor2numpy(tr_data_im)
    y_train = utils_torch.tensor2numpy(tr_data_lab)
    x_test = utils_torch.tensor2numpy(te_data_im)
    y_test = utils_torch.tensor2numpy(te_data_lab)
    #####################

elif args.method[: 6] == 'fmnist':
    sys.path.append('/Tests/Models/blackbox_model_training/image/')
    from Tests.Models.blackbox_model_training.image.train import train_network
    from nn_datasets import load_fashionmnist
    from model import SimpleANN

    dataset_name = 'fmist'
    post_str = ''
    n_classes = 10
    nn_params = {
        # Adversarial Attack Params
        'num_attack_untarg': 500, # number of examples coming from undirected tack on own class trian pts
        'num_attack_targ': 50, # number of examples coming from directed from other class
        'bounds': (0,1), # min/max value bounds for your inputs. default is (0,1) for images. converted to tuple later.
        'attack_norm_untarg': 'LinfPGD',
        'attack_norm_targ': 'LinfPGD',
        'batch_size_adv': 256, # number of adv ex to compute in parallel at a time
        'epsilons': [0.0,0.0002, 0.0005,0.0008, 0.001,0.0015,0.002,0.003,0.01,0.1,0.3,0.5,1.0,], # list of ball sizes in which to search for adv exs
        'eps_use': 'first', # options: first

        # NN Training Params
        'network': 'SimpleANN', 
        'network_dimensions': '700-400-200', 
        'batch_size': 64, 
        'input_dim': 784, 
        'feature_dim': 100, 
        'num_classes': 10, 
        'epochs': 100, 
        'lr': 3, 
        'optimizer': 'SGD', 
        'milestones': [8,15], # lr scaling
        'lr_gamma': 0.5, # lr scaling at each milestone
        'l2_reg': args.l2_reg, # l2 regularization
        'nn_softplus_beta': args.nn_softplus_beta, # softplus beta paramter
    }
    nn_params = utils_io.dict_to_argparse(nn_params)
    nn_params.l2_reg = float(nn_params.l2_reg)
    nn_params.nn_softplus_beta = float(nn_params.nn_softplus_beta)
    #nn_params.l2_reg = np.exp(nn_params.l2_reg)

    # Load Data
    tr_data_im, tr_data_lab, _, tr_loader, _, _, _, _, te_data_im, te_data_lab, _, te_loader = load_fashionmnist(nn_params)

    #####################
    # Train nn

    curdir = './Files/Models/Images/FMNIST-Trained-Model_%s_%s' % (str(nn_params.l2_reg), str(nn_params.nn_softplus_beta))
    utils_io.make_dir(curdir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: %s' % device)
    model = SimpleANN(**nn_params.__dict__).to(device)

    # train model
    if args.nn_retrain == 1:
        model, train_accy, test_accy = train_network(model, nn_params.epochs, tr_loader, te_loader, nn_params.lr, nn_params.optimizer, nn_params.milestones, nn_params.lr_gamma, curdir, nn_params.l2_reg)
        print('train accy: %s' % str(train_accy))
        print('test accy: %s' % str(test_accy))
    else:
        try:
            model.load_state_dict(torch.load(curdir + '/model.pth', map_location = torch.device(device)))
            print('Model Loaded!')
        except:
            warnings.warn('Error in loading saved model. Retraining...')
            model, train_accy, test_accy = train_network(model, nn_params.epochs, tr_loader, te_loader, nn_params.lr, nn_params.optimizer, nn_params.milestones, nn_params.lr_gamma, curdir, nn_params.l2_reg)
            print('train accy: %s' % str(train_accy))
            print('test accy: %s' % str(test_accy))

    # evaluate
    model.eval()
    print('done!')

    from Tests.Models import nn_wrapper
    f_blackbox = nn_wrapper.nn_wrapper(model = model, output_shape = output_shape)
    time_start = time.time()
    ########################################
    # Calculate Manifold Samples
    ########################################

    # assume data is n x d
    if args.kernel == 'WEG' and args.explainer in gpec_explainer_list:
        if args.nn_retrain == 1:
            tmp = decision_boundary.gpec_db_nn(model, tr_data_im, tr_data_lab, tr_loader, nn_params, save = True, save_path = curdir + '/db.pkl')
        else:
            try:
                tmp = utils_io.load_dict(curdir + '/db.pkl')
            except:
                tmp = decision_boundary.gpec_db_nn(model, tr_data_im, tr_data_lab, tr_loader, nn_params, save = True, save_path = curdir + '/db.pkl')
        manifold_samples_dict, geo_matrix_dict, time_manifold = tmp['manifold_samples'], tmp['geo_matrix'], tmp['time']

    x_train = utils_torch.tensor2numpy(tr_data_im)
    y_train = utils_torch.tensor2numpy(tr_data_lab)
    x_test = utils_torch.tensor2numpy(te_data_im)
    y_test = utils_torch.tensor2numpy(te_data_lab)
    #####################


elif args.method[: 7] == 'cifar10':
    sys.path.append('/Tests/Models/blackbox_model_training/image/')
    from Tests.Models.blackbox_model_training.image.train import train_network
    from nn_datasets import load_cifar10
    from model import ResNet18_CIFAR10

    dataset_name = 'cifar10'
    post_str = ''
    n_classes = 10
    nn_params = {
        # Adversarial Attack Params
        'num_attack_untarg': 500, # number of examples coming from undirected tack on own class trian pts
        'num_attack_targ': 50, # number of examples coming from directed from other class
        'bounds': (0,1), # min/max value bounds for your inputs. default is (0,1) for images. converted to tuple later.
        'attack_norm_untarg': 'LinfPGD',
        'attack_norm_targ': 'LinfPGD',
        'batch_size_adv': 256, # number of adv ex to compute in parallel at a time
        'epsilons': [0.0,0.0002, 0.0005,0.0008, 0.001,0.0015,0.002,0.003,0.01,0.1,0.3,0.5,1.0,], # list of ball sizes in which to search for adv exs
        'eps_use': 'first', # options: first

        # NN Training Params
        'network': 'ResNet18_CIFAR10', 
        'batch_size': 32, 
        'epochs': 50, 
        'lr': 0.1, 
        'optimizer': ' adam', 
        'milestones': [25,50,75,100], # lr scaling
        'lr_gamma': 0.5, # lr scaling at each milestone
        'l2_reg': args.l2_reg, # l2 regularization
        'nn_softplus_beta': args.nn_softplus_beta, # softplus beta paramter
    }
    nn_params = utils_io.dict_to_argparse(nn_params)
    nn_params.l2_reg = float(nn_params.l2_reg)
    nn_params.nn_softplus_beta = float(nn_params.nn_softplus_beta)
    #nn_params.l2_reg = np.exp(nn_params.l2_reg)

    # Load Data
    tr_data_im, tr_data_lab, _, tr_loader, _, _, _, _, te_data_im, te_data_lab, _, te_loader = load_cifar10(nn_params)

    #####################
    # Train nn

    curdir = './Files/Models/Images/CIFAR10-Trained-Model_%s_%s' % (str(nn_params.l2_reg), str(nn_params.nn_softplus_beta))
    utils_io.make_dir(curdir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: %s' % device)
    model = ResNet18_CIFAR10(**nn_params.__dict__).to(device)
    
    # train model
    if args.nn_retrain == 1:
        model, train_accy, test_accy = train_network(model, nn_params.epochs, tr_loader, te_loader, nn_params.lr, nn_params.optimizer, nn_params.milestones, nn_params.lr_gamma, curdir, nn_params.l2_reg)
        print('train accy: %s' % str(train_accy))
        print('test accy: %s' % str(test_accy))
    else:
        try:
            model.load_state_dict(torch.load(curdir + '/model.pth', map_location = torch.device(device)))
            print('Model Loaded!')
        except:
            warnings.warn('Error in loading saved model. Retraining...')
            model, train_accy, test_accy = train_network(model, nn_params.epochs, tr_loader, te_loader, nn_params.lr, nn_params.optimizer, nn_params.milestones, nn_params.lr_gamma, curdir, nn_params.l2_reg)
            print('train accy: %s' % str(train_accy))
            print('test accy: %s' % str(test_accy))

    # evaluate
    model.eval()
    print('done!')

    from Tests.Models import nn_wrapper
    f_blackbox = nn_wrapper.nn_wrapper(model = model, output_shape = output_shape)
    ########################################
    # Calculate Manifold Samples
    ########################################

    # assume data is n x d
    if args.kernel == 'WEG' and args.explainer in gpec_explainer_list:
        if args.nn_retrain == 1:
            tmp = decision_boundary.gpec_db_nn(model, tr_data_im, tr_data_lab, tr_loader, nn_params, save = True, save_path = curdir + '/db.pkl')
        else:
            try:
                tmp = utils_io.load_dict(curdir + '/db.pkl')
            except:
                tmp = decision_boundary.gpec_db_nn(model, tr_data_im, tr_data_lab, tr_loader, nn_params, save = True, save_path = curdir + '/db.pkl')
        manifold_samples_dict, geo_matrix_dict, time_manifold = tmp['manifold_samples'], tmp['geo_matrix'], tmp['time']
    
    x_train = utils_torch.tensor2numpy(tr_data_im)
    y_train = utils_torch.tensor2numpy(tr_data_lab)
    x_test = utils_torch.tensor2numpy(te_data_im)
    y_test = utils_torch.tensor2numpy(te_data_lab)
    #####################



# limit train/test samples if specified
x_train, y_train = utils_np.subsample_rows(matrix1 = x_train, matrix2 = y_train, max_rows = args.n_train_samples)
x_test, y_test = utils_np.subsample_rows(matrix1 = x_test, matrix2 = y_test, max_rows = args.n_test_samples)
f_blackbox_predictions = f_blackbox(x_test)

if args.explainer == 'kernelshap' or args.explainer == 'lime' or args.explainer == 'deepshap':
    ##############################################################
    '''
    _____ _____  ______ _____ 
    / ____|  __ \|  ____/ ____|
    | |  __| |__) | |__ | |     
    | | |_ |  ___/|  __|| |     
    | |__| | |    | |___| |____ 
    \_____|_|    |______\_____|
                                
                                
    '''

    ###############################################
    # Generate Explanations
    ###############################################
    time_start = time.time()
    print('=================================')
    print('Generating Explanations...')
    tmp = utils_np.subsample_rows(x_train, 50)
    if args.method == 'mnist' or args.method == 'fmnist' or args.method == 'cifar10':
        # use deepshap for image datasets
        explainer = explainers.deepshap(model, x_train) 
        exp_list = explainer(x_train)
    else:
        if args.explainer == 'kernelshap':
            explainer = explainers.kernelshap(f_blackbox, tmp)
        elif args.explainer == 'lime':
            explainer = explainers.tabularlime(f_blackbox, tmp)
        exp = explainer(x_train)
    print('done!')
    time_explain = time.time() - time_start



    ###############################################
    # Train GP
    ###############################################
    ci_list_multiclass = []
    var_list_multiclass = []
    attr_list_multiclass = []
    time_mean, time_var, time_ci, time_train = 0,0,0,0
    if n_classes == 2: n_classes = 1 # binary prediction
    #### FIX THIS!!
    for target_class in range(n_classes):
    # for target_class in range(1):
        if n_classes >1:
            exp = utils_torch.tensor2numpy(exp_list[target_class])
            if args.kernel == 'WEG':
                geo_matrix = geo_matrix_dict[target_class]
                manifold_samples = manifold_samples_dict[target_class]
                if len(manifold_samples.shape)>2: manifold_samples = manifold_samples.reshape(manifold_samples.shape[0], -1)
        
        
        pred_list = []
        pred_mean_list = []
        pred_var_list = []
        pred_ci_list = []
        data, labels = utils_torch.auto2cuda(x_train), utils_torch.auto2cuda(exp) 
        # data should be n x d
        # labels should be n x d

        max_batch_size = args.max_batch_size # Max number of GP models to train simultaneously
        batched_labels = torch.split(labels, max_batch_size, dim = 1)

        timestamp_label_start = time.time()
        print('=================================')
        print('Training GP... Label %s' % str(target_class))
        print('Number of Batches: %s' % str(len(batched_labels)))

        for i, batch_y in tqdm(enumerate(batched_labels), position = 0, desc = 'Batch Progress'):

            timestamp_start = time.time()
            # reshape data and labels by batch
            batch_size = batch_y.shape[1]
            batch_shape = torch.Size([batch_size])
            tmp_x = data.unsqueeze(0).expand(batch_size,-1,-1)
            tmp_y = batch_y.t()
            # data should be b x n x d
            # labels should be b x n. Features should be in batch dimension.

            # Variance List
            min_var = torch.zeros_like(tmp_y) + 1e-4 # for numerical stability
            tmp_var_list = min_var
            # if args.learn_noise == 1: tmp_var_list = None
            if args.learn_noise == 1:
                learn_addn_noise = True
            else:
                learn_addn_noise = False

            model, likelihood = GP.train_GPEC(tmp_x, tmp_y, manifold_samples, geo_matrix, var_list = tmp_var_list, kernel = args.kernel, n_iter = args.n_iterations, lam = args.lam, rho = args.rho, kernel_normalization = kernel_normalization, batch_shape = batch_shape, lr = args.gpec_lr, learn_addn_noise = learn_addn_noise)

            time_train += time.time() - timestamp_start
            timestamp_start = time.time()

            # Predictions
            model.eval()
            likelihood.eval()
            batch_mean_list = []
            batch_var_list = []
            batch_ci_list = []
            sample_batch = torch.split(utils_torch.numpy2cuda(x_test).float(), args.sample_batch_size, dim = 0)
            for j, batch in enumerate(sample_batch):
                with torch.no_grad():
                    pred = model(batch)
                time_pred = time.time() - timestamp_start
                timestamp_start = time.time()
                
                # mean
                batch_mean_list.append(pred.mean.cpu().detach().numpy())
                time_mean += time.time() - timestamp_start
                timestamp_start = time.time()
                
                # variance
                batch_var_list.append(pred.variance.cpu().detach().numpy())
                time_var += time.time() - timestamp_start
                timestamp_start = time.time()
                
                # confidence interval
                ci_lower = pred.confidence_region()[0].cpu().detach().numpy()
                ci_upper = pred.confidence_region()[1].cpu().detach().numpy()
                batch_ci_list.append(ci_upper - ci_lower)
                time_ci += time.time() - timestamp_start

            pred_mean_list.append(np.concatenate(batch_mean_list, axis = 1))
            pred_var_list.append(np.concatenate(batch_var_list, axis = 1))
            pred_ci_list.append(np.concatenate(batch_ci_list, axis = 1))


        # TODO: Make this able to only calculate uncertainty for certain features (low priority)

        # concatenate predictions
        pred_mean_list = np.concatenate(pred_mean_list, axis = 0).transpose()
        pred_ci_list = np.concatenate(pred_ci_list, axis = 0).transpose()
        pred_var_list = np.concatenate(pred_var_list, axis = 0).transpose()
    
        var_list_multiclass.append(pred_var_list)
        ci_list_multiclass.append(pred_ci_list)
        attr_list_multiclass.append(pred_mean_list)
        print('Done! Time: %s' % str(time.time() - timestamp_label_start))

    if n_classes == 1:
        var_list_multiclass = var_list_multiclass[0]
        ci_list_multiclass = ci_list_multiclass[0]
        attr_list_multiclass = attr_list_multiclass[0]
    


    ###############################################
    # Save Data for Figure
    ###############################################
    saved_data = {
        'n_test_samples': x_test.shape[0],
        'ci_list': ci_list_multiclass,
        'test_predictions': f_blackbox_predictions,
        'rho': args.rho,
        'lam': args.lam,
        'method': args.method,
        'explainer': args.explainer,
        'kernel': args.kernel,
        'time_train': time_train,
        'time_pred': time_pred,
        'time_attr': time_mean,
        'time_var': time_var,
        'time_inference': time_ci + time_pred,
        'time_manifold': time_manifold,
        'time_explain': time_explain,
        'time_ci': time_ci,
        'gamma': args.gamma,
        'l2_reg': args.l2_reg,
        # 'train_accy': train_accy,
        # 'test_accy': test_accy,
    }

    filename = '_'.join([
        args.method,
        args.explainer,
        args.kernel,
        str(args.rho),
        str(args.lam),
        str(args.gamma),
        str(args.l2_reg),
        str(args.nn_softplus_beta),
        ])
    save_path = './Files/Results/regularization_test/%s.pkl' % filename
    print(save_path)
    foldername = os.path.dirname(save_path)
    utils_io.make_dir(foldername)
    utils_io.save_dict(saved_data, save_path)
    # print(ci_list_multiclass.mean())

else:
    ##############################################################
    '''
    _____                                 _                     
    / ____|                               (_)                    
    | |     ___  _ __ ___  _ __   __ _ _ __ _ ___  ___  _ __  ___ 
    | |    / _ \| '_ ` _ \| '_ \ / _` | '__| / __|/ _ \| '_ \/ __|
    | |___| (_) | | | | | | |_) | (_| | |  | \__ \ (_) | | | \__ \
    \_____\___/|_| |_| |_| .__/ \__,_|_|  |_|___/\___/|_| |_|___/
                        | |                                     
                        |_|                                     
    '''

    '''
    if args.explainer == 'bayeslime':

        #f_blackbox = nn_wrapper.nn_wrapper(model = model, output_shape = output_shape, output_type = 'prob')
        f_blackbox.output_type = 'prob'

        sys.path.append('../Bayeslime')
        #import bayeslime.lime_tabular
        from bayeslime.lime_image import LimeImageExplainer
        from bayeslime.lime_tabular import LimeTabularExplainer
        #explainer = bayeslime.lime_tabular.LimeTabularExplainer(x_train, discretize_continuous=False)
        #explainer = LimeImageExplainer(feature_selection = 'none')
        explainer = LimeTabularExplainer(training_data = np.zeros((5,28*28)),feature_selection = 'none')

        interval_list = []
        attr_list = []
        for i,x_sample in tqdm(enumerate(x_test), total = x_test.shape[0]):
            time_start = time.time()
            #x_sample = x_test[i,:].reshape(28,28).astype('double')
            rout = explainer.explain_instance(x_sample.reshape(-1), f_blackbox,
                                    labels=[int(y_test[0])],
                                    num_samples = 200,
                                    model_regressor = 'bayes',
                                    )

            print(time.time() - time_start)
            interval_list.append(rout['blr'].creds)
            attr_list.append(rout['blr'].coef_)
        unc_list = np.array(interval_list)
        attr_list = np.array(attr_list)
        time_inference = time.time() - time_start
        '''
    ### BAYESSHAP
    if args.explainer == 'bayesshap' or args.explainer == 'bayeslime':
        n_features = x_test.shape[1]
        if args.method == 'mnist' or args.method == 'fmnist':
            feature_selection = False # to reduce computation time
            bayesshap_l2 = False
            datatype = 'image'
        else:
            feature_selection = False
            bayesshap_l2 = False
            datatype = 'tabular'

        time_train = 0
        time_start = time.time()

        if args.explainer == 'bayesshap':
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
            if args.bayesshap_idx_start != -1:
                if i < args.bayesshap_idx_start:
                    continue
                if i == args.bayesshap_idx_start + 1:
                    break
            time_start = time.time()
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

            print(time.time() - time_start)
            interval_list.append(rout['blr'].creds)
            attr_list.append(rout['blr'].coef_)
        unc_list = np.array(interval_list)
        attr_list = np.array(attr_list)
        time_inference = time.time() - time_start

    ### CXPLAIN
    elif args.explainer == 'cxplain':
        if args.method == 'mnist' or args.method == 'fmnist' or args.method == 'cifar10':

            time_start = time.time()
            import torch.nn.functional as F
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import sys
            sys.path.append("../cxplain")
            sys.path.append("../cxplain/cxplain")
            from util.test_util import TestUtil
            import tensorflow as tf
            tf.compat.v1.disable_v2_behavior()
            from sklearn.neural_network import MLPClassifier
            from tensorflow.keras.losses import categorical_crossentropy
            from cxplain import UNetModelBuilder, ZeroMasking, CXPlain
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

            if args.method == 'cifar10':
                tmp_x = x_train.reshape(-1, 32,32, 3)
            else:
                tmp_x = x_train.reshape(-1, 28,28, 1)
            tmp_y = utils_torch.tensor2numpy(F.one_hot(utils_torch.numpy2cuda(y_train), num_classes = n_classes))
            downsample_factors = (8,8)
            model_builder = UNetModelBuilder(downsample_factors, num_layers=2, num_units=8, activation="relu",
                                            p_dropout=0.0, verbose=0, batch_size=4, learning_rate=0.001)
            masking_operation = ZeroMasking()
            loss = categorical_crossentropy
            explainer = CXPlain(f_blackbox, model_builder, masking_operation, loss, 
                                num_models=5, downsample_factors=downsample_factors, flatten_for_explained_model=False)
            explainer.fit(tmp_x, tmp_y);

            #############

            time_train = time.time() - time_start

            if args.method == 'cifar10':
                tmp_x = x_test.reshape(-1, 32,32, 3)
            else:
                tmp_x = x_test.reshape(-1, 28,28, 1)

            time_start = time.time()
            attributions, confidence = explainer.explain(tmp_x, confidence_level=0.95)
            print('attribution shape:')
            print(attributions.shape)
            time_inference = time.time() - time_start
        
        else:
            time_start = time.time()
            sys.path.append('../cxplain')
            from tensorflow.keras.losses import categorical_crossentropy
            from cxplain import MLPModelBuilder, ZeroMasking, CXPlain

            from tensorflow.python.framework.ops import disable_eager_execution
            disable_eager_execution()
            import tensorflow as tf
            tf.compat.v1.experimental.output_all_intermediates(True)

            model_builder = MLPModelBuilder(num_layers=2, num_units=24, activation="selu", p_dropout=0.2, verbose=0,
                                            batch_size=8, learning_rate=0.01, num_epochs=250, early_stopping_patience=15)
            masking_operation = ZeroMasking()
            loss = categorical_crossentropy

            explainer = CXPlain(f_blackbox, model_builder, masking_operation, loss, num_models=10)
            explainer.fit(x_train, y_train);

            time_train = time.time() - time_start
            time_start = time.time()

            attributions, confidence = explainer.explain(x_test, confidence_level=0.95)
            time_inference = time.time() - time_start
            # attributions are n x d.
            #confidence is shape n x d x 2. for each sample/feature, ...0 = lower bound, ...1 = upper bound. Calculate Upper - Lower to get width. 

        unc_list = confidence[...,1] - confidence[...,0]
        attr_list = attributions

    ###############################################
    # Save Data for Figure
    ###############################################
    saved_data = {
        'n_test_samples': x_test.shape[0],
        'n_train_samples': x_train.shape[0],
        'ci_list': unc_list,
        'test_predictions': f_blackbox_predictions,
        'method': args.method,
        'explainer': args.explainer,
        'time_train': time_train,
        'time_inference': time_inference,
        'gamma': args.gamma,
        'l2_reg': args.l2_reg,
        # 'train_accy': train_accy,
        # 'test_accy': test_accy,
        'bayesshap_idx_start': args.bayesshap_idx_start,
    }

    if args.explainer in ['bayeslime', 'bayesshap']:

        filename = '_'.join([
            args.method,
            args.explainer,
            args.kernel,
            str(args.rho),
            str(args.lam),
            str(args.gamma),
            str(args.l2_reg),
            str(args.nn_softplus_beta),
            str(args.bayesshap_idx_start),
            ])
    else:
        filename = '_'.join([
            args.method,
            args.explainer,
            args.kernel,
            str(args.rho),
            str(args.lam),
            str(args.gamma),
            str(args.l2_reg),
            str(args.nn_softplus_beta),
            ])
    save_path = './Files/Results/regularization_test/%s.pkl' % filename
    print(save_path)
    foldername = os.path.dirname(save_path)
    utils_io.make_dir(foldername)
    utils_io.save_dict(saved_data, save_path)
    print(unc_list.mean())
