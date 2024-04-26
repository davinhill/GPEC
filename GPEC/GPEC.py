import gpytorch
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import time
import sys
import os

from GPEC import decision_boundary, explainers, GP
from GPEC.utils import *



class GPEC_Explainer():
    '''
    Wrapper function for GPEC explainer. Can be used to calculate explanations (KernelSHAP, LIME, Shapley Sampling, BayesSHAP, BayesLIME, CXPlain) and GPEC uncertainty estimates for a given blackbox model.

    See README.md for more details.
    '''
    def __init__(self,
        f_blackbox,
        x_train,
        y_train = None,
        explain_method = 'kernelshap',
        use_gpec = True,
        kernel = 'WEG',
        lam = 0.5,
        rho = 0.5,
        kernel_normalization = True,
        max_batch_size = 1024,
        max_manifold_samples = None,
        gpec_lr = 0.1,
        gpec_iterations = 30,
        use_labelnoise = True,
        learn_addn_noise = False,
        n_labelnoise_samples = 10,
        labelnoise_var_weighting = 1,
        scale_data = True,
        calc_attr_during_pred = True,

        #------- Explainer-specific parameters
        n_mc_samples = None, # number of samples for bayesshap, bayeslime, and shapley sampling. Will be set to default (1000 for shapley sampling) if not specified.


        #------- Pre-computed Values
        tr_explanations = None,
        tr_explanation_variance = None,
        manifold_samples = None,
        geo_matrix = None,
        **kwargs
        ):
        '''
        args:
            f_blackbox (function): blackbox model to explain. Output must be numpy.
            x_train (numpy matrix): training samples
            y_train (optional, numpy matrix): training labels. Required when using certain explainers
            explain_method (str): Explainer to use. Options: 'kernelshap', 'lime', 'shapleysampling', 'bayesshap', 'bayeslime', 'cxplain'. Default: 'kernelshap'. A custom explainer can by providing pre-calculated explanations using the tr_explanations argument.
            use_gpec (bool): Whether to use GPEC. If False, only the explainer will be used.
            kernel (str): Kernel to use for GPEC. Options: 'WEG' (Weighted Exponential Geodesic), 'RBF' (Radial Basis Function). Default: 'WEG'.
            lam (float): GPEC parameter lambda. Default: 0.5.
            rho (float): GPEC parameter rho. Default: 0.5.
            kernel_normalization (bool): Whether to normalize the kernel similarity such that values are within [0,1] (Eq. 9 in manuscript). Default: True.
            max_batch_size (int): The GP parametrizing GPEC must be trained over all features. This parameter specifies the maximum number of features to train over at once. Default: 1024.
            gpec_lr (float): Learning rate for training GPEC. Default: 0.1.
            gpec_iterations (int): Number of iterations for training GPEC. Default: 30.
            use_labelnoise (bool): Whether to use label noise for GPEC. (Function approximation uncertainty in Eq. 1)
            n_labelnoise_samples (int): If estimating label noise empirically (Eq. 3), specifies number of samples j to use. Default: 10.
            scale_data (bool): Whether to scale the data using StandardScaler. Default: True.
            calc_attr_during_pred (bool): Whether to calculate explanations during prediction (in addition to estimating uncertainty). Default: True.

            n_mc_samples (int): Number of samples for BayesSHAP, BayesLIME, and Shapley Sampling estimation. If not specified, will be set to the default for each explainer.

            tr_explanations (numpy matrix): Pre-calculated explanations for training samples.
            tr_explanation_variance (numpy matrix): Pre-calculated explanation variance (i.e. function approximation uncertainty in Eq. 1).
            manifold_samples (numpy matrix): Pre-calculated decision boundary samples.
            geo_matrix (numpy matrix): Pre-calculated matrix of geodesic distances between the samples in manifold_samples. Calculated by running geo_matrix_kernel() in GPEC.decision_boundary.py on manifold_samples.

        '''

        self.use_gpec = use_gpec
        self.lam = lam
        self.rho = rho
        self.use_labelnoise = use_labelnoise
        self.learn_addn_noise = learn_addn_noise
        self.kernel = kernel
        self.kernel_normalization = kernel_normalization
        self.explain_method = explain_method
        self.f_blackbox = f_blackbox
        self.gpec_lr = gpec_lr
        self.gpec_iterations = gpec_iterations
        self.manifold_samples = manifold_samples
        self.geo_matrix = geo_matrix
        self.n_labelnoise_samples = n_labelnoise_samples
        self.labelnoise_var_weighting = labelnoise_var_weighting
        self.n_mc_samples = n_mc_samples
        self.time_db = {}
        self.explainer = None
        self.scale_data = scale_data
        self.calc_attr_during_pred = calc_attr_during_pred
        if self.scale_data:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)

        if self.use_gpec:
            #==============================
            # Calculate Explanations
            #==============================
            if tr_explanations is None:
                # Calculate Explanations
                timestamp_start = time.time()
                print('=================================')
                print('Generating Explanations...')
                self.attr_list_tr, self.var_list_tr, _ = self._get_explanations(x_train, y_train, x_train, y_train)
                self.time_db['explain_train_samples'] = time.time() - timestamp_start
            else:
                # If explanations are calculated separately
                self.attr_list_tr = tr_explanations
                self.var_list_tr = tr_explanation_variance


            if (kernel == 'WEG') and (self.manifold_samples is None or self.geo_matrix is None):
                #==============================
                # Sample Decision Boundary
                #==============================
                print('=================================')
                print('Sampling Boundary...')
                timestamp_start = time.time()
                if (x_train.shape[1] > 30) and (f_blackbox.model.__class__.__name__ in ['Booster', 'XGBClassifier']):
                    #TODO Implement for NN
                    #TODO implement for >2 classes
                    # select seed samples using adversarial methods
                    seed_samples = decision_boundary.advsamples_xgboost(x_train, y_train, xgb_model = f_blackbox.model,n_samples = 100)
                else:
                    # select initial samples with grid search
                    gridsize_int = int(max(np.exp(-0.2 * (x_train.shape[1]-12)),2))
                    seed_samples = decision_boundary.create_grid_db(x_train, gridsize = gridsize_int)
                self.manifold_samples = decision_boundary.sampledb_DBPS_binary(seed_samples, f_blackbox, decision_threshold = 0.5, n_samples_per_class = 100, batch_size = 4096, max_samples = max_manifold_samples)

                if self.scale_data: self.manifold_samples = self.scaler.transform(self.manifold_samples)
                self.time_db['db_samples'] = time.time() - timestamp_start

                #==============================
                # Build EG Kernel Matrix 
                #==============================
                print('=================================')
                print('Building EG Kernel...')
                timestamp_start = time.time()
                self.geo_matrix = decision_boundary.geo_kernel_matrix(self.manifold_samples)
                self.time_db['geo_matrix'] = time.time() - timestamp_start

            #==============================
            # Train GPEC
            #==============================
            if self.scale_data:
                data = self.scaler.transform(x_train)
            else:
                data = x_train
            data, labels = utils_torch.numpy2cuda(data), utils_torch.numpy2cuda(self.attr_list_tr) # data should be n x d, labels should be n x d
            batched_labels = torch.split(labels, max_batch_size, dim = 1)

            print('=================================')
            print('Training GP...')
            print('Number of Batches: %s' % str(len(batched_labels)))

            self.batched_models = []
            self.batched_likelihoods = []
            timestamp_start = time.time()
            for i, batch_y in tqdm(enumerate(batched_labels), position = 0, desc = 'Batch Progress'): # batched by features

                # reshape data and labels by batch
                batch_size = batch_y.shape[1]
                batch_shape = torch.Size([batch_size])
                tmp_x = data.unsqueeze(0).expand(batch_size,-1,-1)
                tmp_y = batch_y.t()
                # data should be b x n x d
                # labels should be b x n. Features should be in batch dimension.

                # Function Approximation Uncertainty (Label Noise)
                #--------------------------
                min_var = np.zeros_like(tmp_y.cpu()) + 1e-4 # for numerical stability
                if use_labelnoise == 1:
                    tmp_var_list = self.var_list_tr.transpose() * self.labelnoise_var_weighting
                    tmp_var_list = np.maximum(tmp_var_list, min_var)
                else:
                    tmp_var_list = min_var
                # if args.kernel == 'RBF' or args.learn_noise == 1: tmp_var_list = None
                if kernel == 'RBF': tmp_var_list = min_var


                # Train GPEC
                #--------------------------
                # scale GP input
                model, likelihood = GP.train_GPEC(tmp_x, tmp_y, self.manifold_samples, self.geo_matrix, var_list = tmp_var_list, kernel = self.kernel, n_iter = self.gpec_iterations, lam = self.lam, rho = self.rho, kernel_normalization = self.kernel_normalization, batch_shape = batch_shape, lr = self.gpec_lr, learn_addn_noise = self.learn_addn_noise)

                self.batched_models.append(model)
                self.batched_likelihoods.append(likelihood)

                self.time_db['gp_training'] = time.time() - timestamp_start
                print('Training Done!')

        else:

            # NEED TO FIX THIS IN THE FUTURE
            _, _, _ = self._get_explanations(x_train[:1,...], y_train[:1], x_train, y_train)
    
    def explain(self, x, y = None):
        '''
        args:
            x: test data to explain
            y: labels (required for BayesLIME and BayesSHAP explainers)

        return:
            attributions, variance, 95% confidence interval width
        '''
        
        #==============================
        # Calculate Explanations
        #==============================
        if (self.explainer is not None) and (self.calc_attr_during_pred):
            # if using a previously defined explainer
            attr_list, self.var_list_te, ci_list = self._get_explanations(x, y = y)
        else:
            # if the user provided their own pre-calculated explanations
            attr_list, self.var_list_te, ci_list = None, None, None


        if not self.use_gpec:
            # if not using GPEC, directly return explainer output
            return attr_list, self.var_list_te, ci_list

        else:
            #==============================
            # GPEC Uncertainty
            #==============================
            pred_list = []
            # gpec_attr_list = []
            gpec_var_list = []
            gpec_ci_list = []
            if self.scale_data:
                data = self.scaler.transform(x)
            else:
                data = x
            timestamp_start = time.time()
            for i, (model, likelihood) in tqdm(enumerate(zip(self.batched_models, self.batched_likelihoods)), position = 0, desc = 'Batch Progress'): # batched by features
                # Predictions
                #pred_list.append(GP.get_pred(x_test, model, likelihood))
                model.eval()
                likelihood.eval()
                pred_list.append(model(utils_torch.numpy2cuda(data).float()))
                self.time_db['pred'] = time.time() - timestamp_start
                timestamp_start = time.time()
                
                # # mean
                # gpec_attr_list.append(pred_list[i].mean.cpu().detach().numpy())
                # time_mean = time.time() - timestamp_start
                # timestamp_start = time.time()
                
                # variance
                gpec_var_list.append(pred_list[i].variance.cpu().detach().numpy())
                self.time_db['var'] = time.time() - timestamp_start
                timestamp_start = time.time()
                
                # confidence interval
                ci_lower = pred_list[i].confidence_region()[0].cpu().detach().numpy()
                ci_upper = pred_list[i].confidence_region()[1].cpu().detach().numpy()
                gpec_ci_list.append(ci_upper - ci_lower)
                self.time_db['time_ci'] = time.time() - timestamp_start

            # concatenate predictions
            # gpec_attr_list = np.concatenate(gpec_attr_list, axis = 0).transpose()
            gpec_ci_list = np.concatenate(gpec_ci_list, axis = 0).transpose()
            gpec_var_list = np.concatenate(gpec_var_list, axis = 0).transpose()

            return attr_list, gpec_var_list, gpec_ci_list

    def _get_explanations(self, 
        x,
        y = None,
        x_ref = None,
        y_ref = None,
        ):

        if self.explain_method == 'kernelshap':
            attr_list, var_list, ci_list = self._kernelshap(x, x_ref)
        elif self.explain_method == 'lime':
            attr_list, var_list, ci_list = self._lime(x, x_ref)
        elif self.explain_method == 'shapleysampling':
            attr_list, var_list, ci_list = self._shapleysampling(x, x_ref)
        elif self.explain_method == 'bayesshap':
            attr_list, var_list, ci_list = self._bayesshap(x, y, x_ref, kernel = 'shap')
        elif self.explain_method == 'bayeslime':
            attr_list, var_list, ci_list = self._bayesshap(x, y, x_ref, kernel = 'lime')
        elif self.explain_method == 'cxplain':
            attr_list, var_list, ci_list = self._cxplain(x, x_ref, y_ref)

        return attr_list, var_list, ci_list

    def _kernelshap(self, x, x_ref = None):

        if len(self.f_blackbox(x).shape) == 2:
            model = utils_np.binary_mc2sc_modelwrapper(self.f_blackbox)
            # flatten binary classifier output if it's two-dimensional
        else:
            model = self.f_blackbox

        if self.explainer is None: # if no explainer has been previously defined
            self.explainer = explainers.kernelshap(model, x_ref)
        attr_list = self.explainer(x)
        var_list = None
        ci_list = None
        return attr_list, var_list, ci_list

    def _lime(self, x, x_ref = None):
        if self.explainer is None: # if no explainer has been previously defined
            self.explainer = explainers.tabularlime(self.f_blackbox, x_ref, kernel_width = (x_ref.shape[1]**0.5)*0.25)
        attr_list = self.explainer(x)
        var_list = None
        ci_list = None
        return attr_list, var_list, ci_list

    def _shapleysampling(self, x, x_ref = None):
        sys.path.append('../BivariateShapley/BivariateShapley')
        from shapley_sampling import Shapley_Sampling
        # from shapley_datasets import *
        # from utils_shapley import *
        from shapley_explainers import XGB_Explainer

        # Initialize Explainer
        if self.explainer is None: # if no explainer has been previously defined
            #baseline = x_train.mean(axis = 0).reshape(1,-1)
            baseline = 'mean'
            if self.n_mc_samples is None:
                n_mc_samples = 1000
            dataset = pd.DataFrame(x_ref)
            # TODO: Implement this for models other than XGBoost
            self.explainer = XGB_Explainer(pretrained_model = self.f_blackbox.model, baseline = baseline, dataset = dataset, m = n_mc_samples)

        if self.use_labelnoise:
            # Get uncertainty estimate from a standard explainer
            labelnoise_list = []
            for j in range(self.n_labelnoise_samples):
                attr_list = []
                for i,x_sample in tqdm(enumerate(x), total = x.shape[0]):
                    shapley_values, _ = self.explainer(x_sample.reshape(1,-1))
                    attr_list.append(shapley_values)
                attr_list = np.array(attr_list)
                labelnoise_list.append(attr_list)
            labelnoise_list = np.array(labelnoise_list) # m x n x d
        
            var_list = labelnoise_list.var(axis = 0)
            
            if self.n_labelnoise_samples > 50:
                # if there are enough samples, estimate empirically
                ci_list = np.quantile(labelnoise_list, .95, axis = 0) - np.quantile(labelnoise_list, .05, axis = 0)
            else:
                ci_list = (var_list ** 0.5)*2*1.96
        else:
            var_list = None
            ci_list = None
        return attr_list, var_list, ci_list

    def _bayesshap(self, x, y, x_ref = None, kernel = 'shap'):
        if self.explainer is None: # if no explainer has been previously defined
            sys.path.append('../Modeling-Uncertainty-Local-Explainability')
            from bayes.explanations import BayesLocalExplanations, explain_many
            from bayes.data_routines import get_dataset_by_name
            self.explainer = BayesLocalExplanations(training_data=x_ref,
                                                        data="tabular",
                                                        kernel=kernel,
                                                        categorical_features=np.arange(x_ref.shape[1]),
                                                        # discretize_continuous = False,
                                                        verbose=True)
        ci_list = []
        attr_list = []
        var_list = []
        for i,x_sample in tqdm(enumerate(x), total = x.shape[0]):
            rout = self.explainer.explain(classifier_f=self.f_blackbox,
                                    data=x_sample,
                                    label=int(y[0]),
                                    #cred_width=cred_width,
                                    n_samples = self.n_mc_samples,
                                    focus_sample=False,
                                    feature_selection = False,
                                    l2=False)
            ci_list.append(rout['blr'].creds)
            attr_list.append(rout['blr'].coef_)
            var_list.append(rout['blr'].draw_posterior_samples(num_samples = 10000).var(axis = 0))


        ci_list = np.array(ci_list)
        attr_list = np.array(attr_list)
        var_list = np.array(var_list)
        return attr_list, var_list, ci_list

    def _cxplain(self, x, x_ref = None, y_ref = None):
        
        if self.explainer is None: # if no explainer has been previously defined
            sys.path.append('../cxplain')
            from tensorflow.python.keras.losses import categorical_crossentropy
            from cxplain import MLPModelBuilder, ZeroMasking, CXPlain
            # from tensorflow.python.framework.ops import disable_eager_execution
            # disable_eager_execution()
            # import tensorflow as tf
            # tf.compat.v1.experimental.output_all_intermediates(True)
            # tf.compat.v1.disable_v2_behavior()
            # tf.compat.v1.disable_eager_execution()
            # import tensorflow.compat.v1 as tf

            model_builder = MLPModelBuilder(num_layers=2, num_units=24, activation="selu", p_dropout=0.2, verbose=0,
                                            batch_size=8, learning_rate=0.01, num_epochs=250, early_stopping_patience=15)
            masking_operation = ZeroMasking()
            loss = categorical_crossentropy

            self.explainer = CXPlain(self.f_blackbox, model_builder, masking_operation, loss, num_models=10, model_filepath = './Files/Models/tmp_cxplain/')
            self.explainer.fit(x_ref, y_ref)
        attributions, confidence = self.explainer.explain(x, confidence_level=0.95)
        # attributions are n x d.
        #confidence is shape n x d x 2. for each sample/feature, ...0 = lower bound, ...1 = upper bound. Calculate Upper - Lower to get width. 

        ci_list = confidence[...,1] - confidence[...,0]
        attr_list = attributions
        var_list = (ci_list / (2*1.96))**2
        return attr_list, var_list, ci_list
    # def _multiclass_to_singleclass(f_blackbox):
    #     def func(self):
    #         return f_blackbox[]