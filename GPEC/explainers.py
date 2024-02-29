import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')
from GPEC.utils import *

import warnings
try:
    from shap import KernelExplainer, sample, DeepExplainer

    ##############################3
    # KernelSHAP
    ##############################3
    class kernelshap(KernelExplainer):
        def __init__(self, model, train_samples, max_train_samples = 100, **kwargs):
            if train_samples.shape[0] > max_train_samples:
                train_samples = sample(train_samples, max_train_samples)
            super().__init__(model, train_samples, **kwargs)
        def __call__(self, x):
            return self.shap_values(x)


    ##############################3
    # DeepSHAP
    ##############################3

    class deepshap(DeepExplainer):
        def __init__(self, model, train_samples, **kwargs):

            if type(train_samples) == np.ndarray:
                train_samples = utils_torch.numpy2cuda(train_samples)
            super().__init__(model, train_samples)
        def __call__(self, x):
            '''
            returns DeepSHAP explanation

            args:
                x: n x d numpy matrix
            '''
            input_type = type(x)
            if input_type == np.ndarray:
                output = self.shap_values(utils_torch.numpy2cuda(x))
                output = utils_torch.tensor2numpy(output)
            else:
                output = self.shap_values(x)

            return output
except:
    warnings.warn('SHAP is not installed')



##############################3
# LIME
##############################3
try:
    from lime.lime_tabular import LimeTabularExplainer

    class tabularlime(LimeTabularExplainer):
        def __init__(self, model, train_samples, **kwargs):
            super().__init__(train_samples, discretize_continuous=False, **kwargs)
            self.model = model
        def __call__(self, x):
            '''
            returns lime explanation

            args:
                x: n x d numpy matrix
            '''
            explanation_list = []
            for i in tqdm(range(x.shape[0])):
                explanation = self.explain_instance(x[i,:], self.model, num_features=x.shape[1])
                explanation_list.append([i[1] for i in explanation.local_exp[1]])

            return np.array(explanation_list)
except:
    warnings.warn('LIME not installed.')


'''
from tensorflow.python.keras.losses import mean_squared_error
from cxplain import MLPModelBuilder, ZeroMasking, CXPlain

model_builder = MLPModelBuilder(num_layers=2, num_units=24, activation="selu", p_dropout=0.2, verbose=0,
                                batch_size=8, learning_rate=0.01, num_epochs=250, early_stopping_patience=15)
masking_operation = ZeroMasking()
loss = mean_squared_error
'''