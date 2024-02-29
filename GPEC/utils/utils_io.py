import pickle
import os
import numpy as np

def save_dict(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_args(args):
    for x, y in vars(args).items():
        print('{:<16} : {}'.format(x, y))

def chdir_script(file):
    '''
    Changes current directory to that of the current python script

    args:
        file: "__file__"
    '''
    abspath = os.path.abspath(file)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

def get_filedir(file):
    '''
    returns directory path of current file

    args:
        file: "__file__"
    '''
    abspath = os.path.abspath(file)
    dname = os.path.dirname(abspath)
    return dname

def list_fonts():
    '''
    print fonts available in system
    '''
    import matplotlib.font_manager
    fpaths = matplotlib.font_manager.findSystemFonts()

    for i in fpaths:
        f = matplotlib.font_manager.get_font(i)
        print(f.family_name)


def dict_to_argparse(dictionary):
    '''
    converts a dictionary of variables and values to argparse format

    input:
        dictionary
    return:
        argparse object
    '''
    import argparse
    parser = argparse.ArgumentParser()
    for k, v in dictionary.items():
        parser.add_argument('--' + k, default = v)

    args, unknown = parser.parse_known_args()
    return args

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, color_output = 'plotly'):
    '''
    scale matplotlib colormap to set min, mid, and max values.

    output can be a list of rgba colors (for plotly), or a new cmap.
    '''
    from matplotlib import colors

    # shifted index to match the data
    step = (stop - start) / 1000
    max_step = (midpoint - start / 2)
    step = min(step, max_step)

    firsthalf = np.arange(start, midpoint, step)
    secondhalf = np.arange(midpoint, stop+step,step) - midpoint

    shift_index = np.hstack([
        (firsthalf / firsthalf.max())*0.5,
        (secondhalf / secondhalf.max())*0.5+0.5,
        ])

    if color_output == 'plotly':
        color_list = cmap(shift_index).tolist()
        # color_list = ['rgba(' + ','.join(color) + ')' for color in color_list]
        plotly_colorscale = []
        for i,color in enumerate(color_list):
            plotly_colorscale.append(
                # [shift_index[i], 'rgb(' + ','.join(color[:-1]) + ')']
                [i / (len(color_list)-1), colors.rgb2hex(color)]
            )
        return plotly_colorscale
    elif color_output == 'list':
        newcmap = colors.ListedColormap(cmap(shift_index).tolist())
        return newcmap
    elif color_output == 'cmap':
        newcmap = colors.ListedColormap(cmap(shift_index))
        return newcmap