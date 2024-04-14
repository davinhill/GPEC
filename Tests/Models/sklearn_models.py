import numpy as np
import xgboost as xgb
import pickle

class sklearn_wrapper():
    def __init__(self, model_path = None, model = None, output_shape = 'singleclass'):
        '''
        args:
            model_path: model path. either model_path or model required.
            model: xgboost model object. either model_path or model required.
            output_shape: "singleclass" or "multiclass". singleclass returns a vector, multiclass returns a matrix, where each column is the probability of that class.
        '''
        if model_path is None and model is None:
            raise ValueError("Either model_path or model is required")
        if model_path is not None:
            self.model = pickle.load(open(model_path, 'rb'))
        else:
            self.model = model
        self.output_shape = output_shape
        print('done!')
    def __call__(self, x):
        '''
        args:
            x: n x d np matrix
        return:
            prediction
        '''
        if len(x.shape) == 1:
            # x must be a n x d matrix, not a vector.
            x = x.reshape(1,-1)

        output = self.model.predict_proba(x)
        if self.output_shape == 'singleclass':
            return output
        elif self.output_shape == 'multiclass':
            return np.vstack((1-output, output)).transpose()
    def predict_proba(self, x):
        '''
        same as __call__, except only with multiclass (proba) output. For use in some explainers which require predict_proba function.
        '''
        output = self.model.predict_proba(x)
        return np.vstack((1-output, output)).transpose()
