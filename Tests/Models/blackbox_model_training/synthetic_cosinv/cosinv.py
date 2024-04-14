import os

os.chdir('/work/jdy/davin/proj/')


import pandas as pd
import numpy as np
import sys

sys.path.append('./')
from Tests.Models.synthetic_cosinv import *
from GPEC import *

dataset_name = 'cosinv'
prepend = ''
########################################
# SPLIT AND SAVE DATA
########################################
# create a train/test split
f_blackbox = model()
data = data(f_blackbox, n_train = 100, n_test = 10000)
x_train, y_train, x_test, y_test = data.get_data()

np.savetxt('./Files/Data/%s_x_test.csv' % (dataset_name), x_test, delimiter = ',')
np.savetxt('./Files/Data/%s_y_test.csv' % (dataset_name), y_test, delimiter = ',')
np.savetxt('./Files/Data/%s_x_train.csv' % (dataset_name), x_train, delimiter = ',')
np.savetxt('./Files/Data/%s_y_train.csv' % (dataset_name), y_train, delimiter = ',')


########################################
# Load Data
########################################
f_blackbox = model()

x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

########################################
# Calculate Manifold Samples
########################################

# assume data is n x d
#xx_list, grid = decision_boundary.create_grid(x_train, gridsize = 100)
#manifold_samples = decision_boundary.sampledb_plt(xx_list[0], xx_list[1], probs = f_blackbox(grid), decision_threshold = 0)

manifold_samples = decision_boundary.sampledb_func(x_train, f_blackbox)
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
