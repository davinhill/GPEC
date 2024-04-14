import numpy as np

class model():
    '''
    toy model with abs function classifier. Input is 2d.

    '''

    def __init__(self, output_shape = 'singleclass'):
        self.output_shape = output_shape
    def __call__(self, x):
        output = np.abs(x[:,0]) + x[:,1]
        if self.output_shape == 'singleclass':
            return output
        else:
            return np.vstack((1-output, output)).transpose()
    def predict_proba(self, x):
        output = np.abs(x[:,0]) + x[:,1]
        return np.vstack((1-output, output)).transpose()



    def db(self, x):
        '''
        returns samples from the decision boundary

        input:
            x: 1d numpy vector

        return:
            np vector of dim n x 2
        '''
        y = -np.abs(x)
        return np.vstack((x,y)).transpose()

class data():
    '''
    generate synthetic data for toy models
    '''

    def __init__(self, toy_model, n_train = 5000, n_test = 2000, seed = 0):
        np.random.seed(seed)
        low = -15
        high = 15
        x = np.random.uniform(low = low, high = high, size = (n_train+n_test, 2))
        y = (toy_model(x)>0)*1

        self.x_train = x[:n_train,:]
        self.x_test = x[n_train:,:]
        self.y_train = y[:n_train]
        self.y_test = y[n_train:]

        # # create grid
        # xmin, xmax = low,high
        # ymin, ymax = low,high
        # int_x = (xmax-xmin) / (n_train**0.5)
        # int_y = (ymax-ymin) / (n_train**0.5)
        # xx, yy = np.mgrid[xmin:xmax+int_x:int_x, ymin:ymax+int_y:int_y]
        # grid = np.c_[xx.ravel(), yy.ravel()]
        # self.x_train = grid
        # self.y_train = (toy_model(self.x_train)>0)*1

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
