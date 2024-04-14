import numpy as np
import torch
import sys
sys.path.append('./')
from GPEC.utils.utils_torch import *

class nn_wrapper():
    def __init__(self, model_path = None, model = None, output_shape = 'multiclass', output_type = 'default', target_class = 0, unflatten_image = False, color_dim = 3, chw = False):
        '''
        args:
            model_path: model path. either model_path or model required.
            model: xgboost model object. either model_path or model required.
            output_shape: "singleclass" or "multiclass". singleclass returns a vector, multiclass returns a matrix, where each column is the probability of that class.
            target_class: if output shape is singleclass, then target class must be provided to generate one vs all probability.
            unflatten_image: if the input data is n x d, unflatten to n x (d / color_dim) **0.5 x (d / color_dim) **0.5 x color_dim. Assumes square images.
        '''
        if model_path is None and model is None:
            raise ValueError("Either model_path or model is required")
        if model_path is not None:
            raise ValueError('Not Implemented Yet.')
        else:
            self.model = model
        self.output_shape = output_shape
        self.target_class = target_class
        self.output_type = output_type
        self.unflatten_image = unflatten_image
        self.color_dim = color_dim
        self.eval = self.model.eval
        self.chw = chw
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
        elif self.chw == True and len(x.shape) > 2 and x.shape[1] > 3:
            # reshape to n x c x h x w instead of n x h x w x c
            x = x.reshape(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        
        if self.unflatten_image and (len(x.shape) == 2):
            n, d = x.shape
            h = int((d // self.color_dim) ** 0.5)
            x = x.reshape(n, self.color_dim, h, h)

        input_type = type(x)
        if input_type == np.ndarray:
           x = numpy2cuda(x) 

        self.model.eval()
        output = self.model(x.type(dtype=torch.float32))
        if self.output_type == 'prob':
            output = F.softmax(output, dim = 1)
        if input_type == np.ndarray:
           output = tensor2numpy(output)

        if self.output_shape == 'singleclass':
            return output[:,self.target_class]
        elif self.output_shape == 'multiclass':
            return output
                
    def predict(self, x):
        '''
         Returns prediction as a vector. For use in some explainers which require predict function.
        '''
        # input_type = type(x)
        # if input_type == np.ndarray:
        #    x = numpy2cuda(x) 
        # else:
        #     device = x.device
        #     x = tensor2cuda(x)

        # output = self.model(x.type(dtype=torch.float32)).argmax(dim = 1)
        # output = F.one_hot(output, num_classes = 10)
        # if input_type == np.ndarray:
        #    output = tensor2numpy(output)
        # else:
        #     output = output.to(device)
        # return output
        return self.__call__(x)