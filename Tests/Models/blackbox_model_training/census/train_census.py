# adapted from https://github.com/slundberg/shap

import os
os.chdir('/work/jdy/davin/proj/')


import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys
import shap

sys.path.append('./')
from GPEC import *
from GPEC.utils import *

np.random.seed(1)

dataset_name = 'census'
post_str = ''
X,y = shap.datasets.adult()
utils_io.save_dict(X.columns.tolist(), './Files/Data/%s_colnames.pkl' % dataset_name)
X = X.values

# create a train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

d_train = xgboost.DMatrix(x_train, label=y_train)
d_test = xgboost.DMatrix(x_test, label=y_test)

np.savetxt('./Files/Data/%s_x_test.csv' % (dataset_name), x_test, delimiter = ',')
np.savetxt('./Files/Data/%s_y_test.csv' % (dataset_name), y_test, delimiter = ',')
np.savetxt('./Files/Data/%s_x_train.csv' % (dataset_name), x_train, delimiter = ',')
np.savetxt('./Files/Data/%s_y_train.csv' % (dataset_name), y_train, delimiter = ',')



colnames = ['Age', 'Workclass', 'Education-Num', 'Marital Status', 'Occupation',
       'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
       'Hours per week', 'Country']

d_train = xgboost.DMatrix(x_train, label=y_train)
d_test = xgboost.DMatrix(x_test, label=y_test)

params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss"
}

model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

pred =(model.predict(d_train) >= 0.5)*1
print('train accy: %s' % str(metrics.accuracy_score(y_train, pred)))

pred =(model.predict(d_test) >= 0.5)*1
print('test accy: %s' % str(metrics.accuracy_score(y_test, pred)))

model.save_model('./Files/Models/model_%s%s.json' % (dataset_name, post_str))
print('done!')



########################################
# Load Data
########################################
from Tests.Models import xgb_models
model_path = './Files/Models/model_%s%s.json' % (dataset_name, post_str)
f_blackbox = xgb_models.xgboost_wrapper(model_path, output_shape = 'singleclass')

x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

########################################
# Calculate Manifold Samples
########################################

# assume data is n x d
xx_list, grid = decision_boundary.create_grid(x_train, gridsize = 2)
manifold_samples = decision_boundary.sampledb_DBPS_binary(grid, f_blackbox, decision_threshold = 0.5, n_samples_per_class = 100, batch_size = 4096)
geo_matrix = decision_boundary.geo_kernel_matrix(manifold_samples)

'''
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x = manifold_samples[:,0], y = manifold_samples[:,1])
plt.savefig('./fig.jpg')
'''

# Save
post_string = ''
np.save('./Files/Models/%s_samples%s.npy' % (dataset_name, post_string), manifold_samples)
np.save('./Files/Models/%s_geomatrix%s.npy' % (dataset_name, post_string), geo_matrix)

print('done!')

