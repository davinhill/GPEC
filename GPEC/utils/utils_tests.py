import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import os
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


from GPEC.utils import utils_io


def uncertaintyplot(x_train, x_test, hue_list, save_path,f_blackbox = None, feat_list = [0], cmap = sns.cubehelix_palette(as_cmap=True), decision_threshold = 0.5, lam = None, rho = None, plot_train = True, center_cmap = False, center = 0, cmap_scaling = 1, alpha = 1, axislim = None):
    '''
    uncertainty plot for testing purposes + sensitivity plots
    
    '''
    if axislim is None:
        xmin, xmax, ymin, ymax = x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max()
    else:
        xmin, xmax, ymin, ymax = axislim
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Variance
    fig, axes = plt.subplots(1, len(feat_list), figsize=(6 * len(feat_list), 3), sharey=False, sharex = False, squeeze = False)

    for i,feat in enumerate(feat_list):
        '''
        if center_cmap:
            normalize = mcolors.TwoSlopeNorm(vcenter=center, vmin=hue_list[:,feat].min(), vmax=hue_list[:,feat].max()*cmap_scaling)
        else:
            normalize = mcolors.TwoSlopeNorm(vmin=hue_list[:,feat].min(), vcenter = (hue_list[:,feat].max()*cmap_scaling - hue_list[:,feat].min())/2 + hue_list[:,feat].min(), vmax=hue_list[:,feat].max()*cmap_scaling)

        sns.scatterplot(x=x_test[:,0], y=x_test[:,1], c = hue_list[:,feat], ax = axes[0,i], cmap = cmap, norm = normalize, legend = False, marker = 's', s = 50, edgecolor=None, alpha = alpha)
        #sns.scatterplot(x=x_test[:,0], y=x_test[:,1], c = hue_list[:,feat], ax = axes, cmap = cmap, norm = normalize, legend = False, marker = 's', s = 50, edgecolor=None)

        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(hue_list[:,feat])
        divider = make_axes_locatable(axes[0,i])
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fmt = lambda x, pos: '{:.2f}'.format(x) # formatting
        fig.colorbar(scalarmappaple, cax=cax, orientation='vertical', format=FuncFormatter(fmt))
        '''
        if center_cmap:

            normalize = mcolors.TwoSlopeNorm(vcenter=center, vmin=hue_list[:,i].min(), vmax=hue_list[:,i].max())
            sns.scatterplot(x=x_test[:,0], y=x_test[:,1], c = hue_list[:,i], ax = axes[0,i], cmap = cmap, norm = normalize, legend = False)

            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(hue_list[:,i])
            fig.colorbar(scalarmappaple)

        else:
            normalize = mcolors.TwoSlopeNorm(vmin=hue_list[:,i].min(), vcenter = hue_list[:,i].mean(), vmax=hue_list[:,i].max())
            sns.scatterplot(x=x_test[:,0], y=x_test[:,1], c = hue_list[:,i], ax = axes[0,i], cmap = cmap, norm = normalize, legend = False, s = 50, edgecolor=None)

            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(hue_list[:,i])
            fig.colorbar(scalarmappaple)

        # Decision Boundary
        if f_blackbox is not None:
            probs = f_blackbox(grid)
            if len(probs.shape)>1: probs = probs[:,0]
            probs = probs.reshape(xx.shape)

            cmap_single = mcolors.LinearSegmentedColormap.from_list("", ["Chartreuse", "White"])
            axes[0,i].contour(xx, yy, probs, levels=[decision_threshold], cmap = cmap_single, vmin=-0.6, vmax=.1, linewidths=9, alpha = 0.9, zorder = 4)
            cmap_single = mcolors.LinearSegmentedColormap.from_list("", ["Chartreuse", "Black"])
            axes[0,i].contour(xx, yy, probs, levels=[0.5], cmap = cmap_single, vmin=-0.6, vmax=.1, linewidths=5, alpha = 1, zorder = 5)

        axes[0,i].set_xlim((xmin,xmax))
        axes[0,i].set_ylim((ymin,ymax))
        axes[0,i].set_title('Lambda: %s, Rho: %s' % (str(lam), str(rho)))

        # Train Points
        if plot_train:
            sns.scatterplot(x=x_train[:,0], y=x_train[:,1], color = 'red', s=100, ax = axes[0,i], zorder = 10, alpha = 0.9)

        #axes[0,i].legend(bbox_to_anchor=(1.15, 0.5), loc = 'center right')


    foldername = os.path.dirname(save_path)
    utils_io.make_dir(foldername)
    plt.savefig(save_path, format='jpg')
    print(save_path)


def plot_hue(x_test, hue_list, cmap, axes, center_cmap = False, center = 0, feat = 0, alpha = 1, cmap_scaling = 1, cbar_range = None):
    '''
    plot uncertainty plots (final plots for paper)
    '''

    if cbar_range is None:
        if center_cmap:
            vcenter = center
            vmin = hue_list[:,feat].min()
            vmax = hue_list[:,feat].max()*cmap_scaling

            if vmin >= center:
                vmin = -vmax
        else:
            vcenter = (hue_list[:,feat].max()*cmap_scaling - hue_list[:,feat].min())/2 + hue_list[:,feat].min()
            vmin = hue_list[:,feat].min()
            vmax = hue_list[:,feat].max()*cmap_scaling
    else:
        vmin, vcenter, vmax = cbar_range
    if center_cmap:
        normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    else:
        normalize = mcolors.TwoSlopeNorm(vmin=vmin, vcenter = vcenter, vmax=vmax)

    sns.scatterplot(x=x_test[:,0], y=x_test[:,1], c = hue_list[:,feat], ax = axes, cmap = cmap, norm = normalize, legend = False, marker = 's', s = 50, edgecolor=None, alpha = alpha)
    #sns.scatterplot(x=x_test[:,0], y=x_test[:,1], c = hue_list[:,feat], ax = axes, cmap = cmap, norm = normalize, legend = False, marker = 's', s = 50, edgecolor=None)

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
    scalarmappaple.set_array(hue_list[:,feat])
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    fmt = lambda x, pos: '{:.1f}'.format(x) # formatting
    cbar = plt.colorbar(scalarmappaple, cax=cax, orientation='vertical', format=FuncFormatter(fmt))

    # from matplotlib import ticker
    # # ticklabels = np.arange(vmin, vmax, 0.05).round(5).tolist()
    # ticklabels = [0.0, 0.01,0.1, 0.2]
    # cbar.set_ticks(ticklabels)
    # cbar.set_ticklabels(ticklabels)