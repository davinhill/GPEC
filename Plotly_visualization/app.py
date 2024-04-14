# Code adapted from https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-svm

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from matplotlib import colormaps
import sys
import os
sys.path.append('../')
from GPEC import *
from GPEC.utils import * 
from Tests.Models import sklearn_models

import utils.dash_reusable_components as drc
import utils.figures as figs

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "GPEC: Geodesic Distance"



def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)

    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )
    elif dataset == "linear":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )

class f_db_linear():
    def __init__(self,model, data_transform):
        self.model = model
        self.data_transform = data_transform
    
    def __call__(self, x0_samples):
        coef = self.model.coef_
        bias = self.model.intercept_

        if len(x0_samples.shape) == 1:
            x0_samples = x0_samples.reshape(-1,1)
        x1_samples = np.hstack((x0_samples, np.zeros_like(x0_samples)))
        x1_samples = - (1/coef[0,1]) * (self.data_transform(x1_samples) @ coef.transpose() + bias)
        return x1_samples

class poly_transform_x0():
    def __init__(self, power):
        self.power = power
    def __call__(self, x):
        return np.hstack((x, np.power(x[:,0:1], np.arange(2,self.power+1))))

def WEG(x,y, rho, manifold_samples, EG_kernel):
    # Weight EG Kernel Matrix

    # l2 distance weighting
    dist_x = distances.l2_parallel(x, manifold_samples)
    dist_x = utils_np.exp_kernel_func(dist_x, lam = rho, q = 2)
    dist_x = dist_x / dist_x.sum(axis = -1, keepdims = True)
    if np.array_equal(x,y):
        x_eq_y = True
        dist_y = dist_x
    else:
        x_eq_y = False
        dist_y = distances.l2_parallel(y, manifold_samples)
        dist_y = utils_np.exp_kernel_func(dist_y, lam = rho, q = 2)
        dist_y = dist_y / dist_y.sum(axis = -1, keepdims = True)

    output = dist_x @ EG_kernel @ dist_y.transpose()

    if x_eq_y:
        n1 = n2 = output
    else:
        # normalization
        n1 = dist_x @ EG_kernel @ dist_x.transpose()
        n2 = dist_y @ EG_kernel @ dist_y.transpose()
    n1 = n1.diagonal() ** 0.5 # n-dimensional vector
    n2 = n2.diagonal() ** 0.5 # n-dimensional vector
    normfactor = np.outer(n1, n2) # n x n
    output = np.divide(output, normfactor)

    
    return output


app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    style=dict(display='flex'),
                    children=[
                        dcc.Markdown('# GPEC Exploration Tool')
                    ],
                )
            ],
        ),
        html.Div(
            className = "container scalable",
            children = [
                ##################################
                # SECTION 1 TEXT
                ##################################
                dbc.Row([
                    dbc.Col(width = dict(size = 1, order = 'first')),
                    dbc.Col(width = dict(size = 1, order = 'last')),
                    dbc.Col(
                        children = [
                            dcc.Markdown('''

                            ## 1. Geodesic Distance and Model Complexity

                            #### Background

                            The first step in defining a decision boundary-based uncertainty is to decide how to quantify the boundary's complexity. Let's imagine a simple 2d classifier and consider a section of its decision boundary between two points A and B. Intuitively, the more that this section of decision boundary "changes" between A and B, the more complex the model is.

                            We can measure this "change" in decision boundary using geodesics. Geodesics are length-minimizing paths that are constrained to an underlying manifold. By viewing the decision boundary as a manifold, we can calculate geodesic distances between samples on the decision boundary.

                            In the visualization below, we take samples on the decision boundary and explore how the geodesic distances between points changes as the decision boundary changes. Notice that geodesic distances are minimized by linear (k=1) models. We can see that geodesic distance reflects the local complexity of the decision boundary between any two given points.

                            #### Interact

                            The top plot shows logistic regression classifier with polynomial transformation of the x-axis variable. We take samples along the model decision boundary and calculate geodesic distances between the sample points which are shown in the bottom two heatmaps. The right heatmap is for reference; the values are calculated for a logistic regression with no transformation, which represents the simplest possible model.

                            **Dataset**: Select the Scikit-Learn toy dataset to model  
                            **Classifier Polynomial Power**: The polynomial degree for the logistic regression classifier can be selected from this slider.  
                            **Decision Boundary Samples**: Select the number of decision boundary samples.  
                            **Distance Comparison**: Choose the heatmap (bottom right) which serves as a reference point for the selected model. The reference model is a logistic regression model with polynomial degree = 1.  

                            &nbsp;

                            ''', mathjax = True),
                        ]
                    ),
                ]),
                ##################################
                # SECTION 1 Interact
                ##################################
                dbc.Row([
                    dbc.Col(width = dict(size = 1, order = 'first')),
                    dbc.Col(width = dict(size = 1, order = 'last')),
                    dbc.Col(
                        width = dict(size=2),
                        children = [
                        ##################################
                        # SECTION 1 Toggles
                        ##################################
                            drc.NamedDropdown(
                                name="Select Dataset",
                                id="dropdown-select-dataset",
                                options=[
                                    {"label": "Moons", "value": "moons"},
                                    {
                                        "label": "Linearly Separable",
                                        "value": "linear",
                                    },
                                    {
                                        "label": "Circles",
                                        "value": "circles",
                                    },
                                ],
                                clearable=False,
                                searchable=False,
                                value="moons",
                            ),
                            drc.NamedSlider(
                                name="Classifier Polynomial Power",
                                id="slider-polynomial-K",
                                min=1,
                                max=10,
                                value=5,
                                step = 1,
                                marks={
                                    i : str(i)
                                    for i in range(0, 11, 1)
                                },
                            ),
                            drc.NamedSlider(
                                name="Decision Boundary Samples",
                                id="slider-db-samples",
                                min=2,
                                max=20,
                                step = 1,
                                value=10,
                                marks={
                                    i : str(i)
                                    for i in range(2, 21, 2)
                                },
                            ),
                            drc.NamedDropdown(
                                name="Select Distance Comparison",
                                id="dropdown-select-comparison",
                                options=[
                                    {"label": "Linear Model", "value": "linear"},
                                    {
                                        "label": "Linear Model Ratio",
                                        "value": "linear_ratio",
                                    },
                                    {
                                        "label": "Euclidean",
                                        "value": "euclidean",
                                    },
                                    {
                                        "label": "Euclidean Ratio",
                                        "value": "euclidean_ratio",
                                    },
                                ],
                                clearable=False,
                                searchable=False,
                                value="linear_ratio",
                            ),
                            ]
                        ),

                    ##################################
                    # SECTION 1 Plots
                    ##################################
                    dbc.Col(
                        children=[
                            dcc.Loading(
                            className="graph-wrapper",
                            # style={'width': '90vh', 'height': '90vh'},
                            children=dcc.Graph(id="graph-sklearn-top"),
                            # style={"display": "none"},
                            ),
                            ###########################
                            # Heatmaps Row 1
                            html.Div(
                                style=dict(display='flex'),
                                # style=dict(width = '20%', display='flex-wrap'),
                                className='six column',
                                id = '2nd-row-graphs',
                                children = [
                                    html.Div(
                                        style=dict(width = '50%'),
                                        # className = 'six column',
                                        children = [
                                            dcc.Loading(
                                                # className="graph-wrapper",
                                                children=dcc.Graph(id="heatmap1"),
                                            ),
                                            ]
                                        ),
                                    html.Div(
                                        style=dict(width = '50%'),
                                        # className = 'six column',
                                        children = [
                                            dcc.Loading(
                                                # className="graph-wrapper",
                                                children=dcc.Graph(id="heatmap2"),
                                            ),
                                            ]
                                        ),
                                ],
                            ),
                        ]
                    )
                    ]),

                
                ##################################
                # SECTION 2 TEXT
                ##################################
                dbc.Row([
                    dbc.Col(width = dict(size = 1, order = 'first')),
                    dbc.Col(width = dict(size = 1, order = 'last')),
                    dbc.Col(
                        children=[
                                dcc.Markdown('''

                                &nbsp;

                                '''),

                                html.Hr(style={'borderWidth': "0.1vh", "width": "100%", "borderColor": "#A0A1A1", "borderStyle":"solid", "opacity": "80%"}),

                                dcc.Markdown('''

                                ## 2. Weighted Exponential Geodesic Kernel

                                #### Background

                                We next want to define a kernel similarity measure between two sample explanations $E_1,E_2$ that incorporates the geodesic distances calculated between decision boundary samples. We can do this by first taking the Exponential Geodesic (EG) Kernel on the boundary samples, and then weighting each geodesic segment based on their $L_2$ distances to $E_1$ and $E_2$. This forms the Weighted Exponential Geodesic (WEG) Kernel.

                                &nbsp;

                                **EG Kernel**  
                                
                                We apply an exponential weighting to the geodesic distance matrix from Section 1:
                                $K_{\\textrm{EG}}(X_i, X_j) = \\exp[-\\lambda d_{geo}(X_i, X_j)]$. The resulting heatmap is showin the bottom left. The parameter $\\lambda$ controls how we want to weight the boundary complexity in the similarity calculation. Lower values of $\\lambda$ means that decision boundary complexity has more of an effect on the WEG kernel similarity between $E_1$ and $E_2$.

                                &nbsp;

                                **Weighting Matrix**  
                                We define a weighting matrix $W$, which is represented by the heatmap on the bottom right. Each element $W_{i,j}$ is defined as $W_{i,j} = \\exp[-\\rho (||X_i - E_1||^2_2 + ||X_j - E_2||^2_2)]$. The parameter $\\rho$ controls how large a neighborhood we want to consider when calculating the similarity between $E_1$ and $E_2$. As $\\rho$ decreases, we more heavily weight the boundary points closest to $E_1$ and $E_2$, i.e. we more consider the boundary complexity in a relatively small neighborhood. Conversely, as $\\rho$ increases, all the boundary points become weighted more evenly.


                                &nbsp;

                                We can then calculate the WEG kernel similarity by multiplying and then summing the two matrices: $K_{\\textrm{WEG}}(E1, E2)= \sum_{i,j} K_{\\textrm{EG}}(X_i, X_j) * W_{ij}$. Intuitively, we are taking the EG kernel matrix (left matrix) and applying weights to each element (right matrix).

                                #### Interact
                                In the figure below, we plot the two explanation samples $E_1$ and $E_2$ and the dotted line segments conecting $E_1$ and $E_2$ to the boundary samples. The line segment width represents the $L_2$ weighting. We can select $\\rho$ and $\\lambda$ parameters using the sliders on the left. As you change these parameters, notice how the values in the two heatmaps change.

                                &nbsp;

                                ''', mathjax = True),

                        ]
                    )

                ]),
                ##################################
                # SECTION 2 INTERACT
                ##################################
                dbc.Row([
                    dbc.Col(width = dict(size = 1, order = 'first')),
                    dbc.Col(width = dict(size = 1, order = 'last')),
                    ##################################
                    # SECTION 2 Toggles
                    ##################################
                    dbc.Col(
                        width=2,
                        children=[
                            drc.NamedSlider(
                                name="Lambda",
                                id="slider-lambda",
                                min=-2,
                                max=0,
                                # step = 1,
                                value=-1,
                                marks={i: '{}'.format(10 ** i) for i in range(-2, 1)},
                            ),
                            drc.NamedSlider(
                                name="Rho",
                                id="slider-rho",
                                min=-2,
                                max=0,
                                # step = 1,
                                value=-1,
                                marks={i: '{}'.format(10 ** i) for i in range(-3, 1)},
                            ),
                    
                    ]),
                    ##################################
                    # SECTION 2 Plots
                    ##################################
                    dbc.Col(
                        children=[
                            html.Div(
                                # style = dict(display = 'flex'),
                                className="six column",
                                id = 'section2-plot',
                                children = [

                                    dcc.Loading(
                                    # style={'width': '90vh', 'height': '90vh'},
                                    children=dcc.Graph(id="graph-plot2"),
                                    # style={"display": "none"},
                                    ),
                                ]
                            ),

                            html.Div(
                                style=dict(display='flex'),
                                # style=dict(width = '20%', display='flex-wrap'),
                                className='six column',
                                id = 'section2-row2graphs',
                                children = [
                                    html.Div(
                                        style=dict(width = '50%'),
                                        # className = 'six column',
                                        children = [
                                            dcc.Loading(
                                                # className="graph-wrapper",
                                                children=dcc.Graph(id="heatmap3"),
                                            ),
                                            ]
                                        ),
                                    html.Div(
                                        style=dict(width = '50%'),
                                        # className = 'six column',
                                        children = [
                                            dcc.Loading(
                                                # className="graph-wrapper",
                                                children=dcc.Graph(id="heatmap4"),
                                            ),
                                            ]
                                        ),
                                ],
                            ),
                        ]
                    )
                ]),

                ##################################
                # SECTION 3 TEXT
                ##################################
                dbc.Row([
                    dbc.Col(width = dict(size = 1, order = 'first')),
                    dbc.Col(width = dict(size = 1, order = 'last')),
                    dbc.Col(
                        children=[
                                dcc.Markdown('''

                                &nbsp;

                                '''),

                                html.Hr(style={'borderWidth': "0.1vh", "width": "100%", "borderColor": "#A0A1A1", "borderStyle":"solid", "opacity": "80%"}),

                                dcc.Markdown('''

                                ## 3. GPEC Uncertainty Estimate

                                #### Background

                                With the WEG kernel defined, we can then train the Gaussian Process model that parametrizes GPEC. We use the two explanation samples $E_1$ and $E_2$ as labels and calculate the WEG kernel matrix. To estimate the GPEC uncertainty of a test explanation $E*$, we calculate the variance of the predictive distribution for that sample.

                                #### Interact

                                The figure below shows the contour lines for GPEC uncertainty estimates calculated for a grid of explanations. Using the slider on the left, you can control the estimated function approximation uncertainty, which represented by explanation variance. Note that as this explanation variance increases, the overall uncertainty estimate for GPEC also increases.

                                &nbsp;

                                '''),

                        ]
                    )

                ]),
                ##################################
                # SECTION 3 INTERACT
                ##################################
                dbc.Row([
                    dbc.Col(width = dict(size = 1, order = 'first')),
                    dbc.Col(width = dict(size = 1, order = 'last')),
                    ##################################
                    # SECTION 3 Toggles
                    ##################################
                    dbc.Col(
                        width=2,
                        children=[
                            drc.NamedSlider(
                                name="Explanation Variance",
                                id="slider-noise",
                                min=0,
                                max=1,
                                # step = 1,
                                value=0,
                                marks={
                                    i / 10: str(i / 10)
                                    for i in range(0, 11, 2)
                                },
                            ),
                    ]),
                    ##################################
                    # SECTION 3 Plots
                    ##################################
                    dbc.Col(
                        children=[
                            html.Div(
                                # style = dict(display = 'flex'),
                                className="six column",
                                id = 'section3-plot',
                                children = [
                                    dcc.Loading(
                                    # style={'width': '90vh', 'height': '90vh'},
                                    children=dcc.Graph(id="graph-plot3"),
                                    # style={"display": "none"},
                                    ),
                                ]
                            ),
                        ]
                    )
                ]),

            ]
        ),  
    ]
)



@app.callback(
    [
        Output("graph-sklearn-top", "figure"),
        Output("heatmap1", "figure"),
        Output("heatmap2", "figure"),
        Output("graph-plot2", "figure"),
        Output("heatmap3", "figure"),
        Output("heatmap4", "figure"),
        Output("graph-plot3", "figure"),
    ],
    [
        ### Plot 1
        Input("dropdown-select-dataset", "value"),
        # Input("slider-dataset-noise-level", "value"),
        Input("slider-polynomial-K", "value"),
        Input("slider-db-samples", "value"),
        Input("dropdown-select-comparison", "value"),
        ### only for updating contours upon zoom
        Input('graph-sklearn-top', 'relayoutData'),
        State('graph-sklearn-top', 'figure'),
        ### Plot 2
        Input("slider-rho", "value"),
        Input("slider-lambda", "value"),
        ### Plot 3
        Input("slider-noise", "value"),
    ],
)
def update_graph(
    dataset,
    # noise,
    poly_power,
    n_db_samples,
    comparison,
    relayout_data,
    figure,
    rho,
    lam,
    exp_var,
):
    

    rho = 10 ** rho
    lam = 10 ** lam
    noise = 0.2
    sim_points = np.array([[-0.5, 2], [0.5, 2]])
    # exp_var = 0.1
    threshold = 0.5
    sample_size = 300
    # poly_power = 5
    # n_db_samples = 10

    # Data Pre-processing
    X, y = generate_data(n_samples=sample_size, dataset=dataset, noise=noise)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    ##### Change plot area depending on zoom level
    try:
        # case: manual zoom
        x_min = relayout_data['xaxis.range[0]']
        x_max = relayout_data['xaxis.range[1]']
        y_min = relayout_data['yaxis.range[0]']
        y_max = relayout_data['yaxis.range[1]']

        xint = x_max - x_min
        yint = y_max - y_min
        h_y = yint / 1000
        h_x = xint / 1000

    except (KeyError, TypeError):
        try:
            # case: autoscale
            if relayout_data['xaxis.autorange']==True:
                [x_min, x_max] = figure['layout']['xaxis']['range']
                [y_min, y_max] = figure['layout']['yaxis']['range']
                xint = x_max - x_min
                yint = y_max - y_min
                h_y = yint / 1000
                h_x = xint / 1000
                # maxint = max(x_max - x_min, y_max - y_min)
                # h = maxint / 1000 # step size in mesh

        except (KeyError, TypeError):
            # case: initializing plot
            x_min = X[:, 0].min() - 0.5
            x_max = X[:, 0].max() + 0.5
            y_min = X[:, 1].min() - 0.5
            y_max = X[:, 1].max() + 0.5
            h_x = h_y = 0.04  # step size in the mesh
    #############

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Train Model
    data_transform = poly_transform_x0(power = poly_power)
    X_train_tf = data_transform(X_train)
    X_test_tf = data_transform(X_test)
    clf = LogisticRegression(penalty = 'none')
    clf.fit(X_train_tf, y_train)
    f_db = f_db_linear(clf, data_transform)

    Z = (clf.predict_proba(data_transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1] > 0.5)*1


    # uncertainty test grid
    x_min_test = X[:, 0].min() - 10
    x_max_test = X[:, 0].max() + 10
    y_min_test = X[:, 1].min() - 10
    y_max_test = X[:, 1].max() + 10
    h_x_test = h_y_test = 0.5  # step size in the mesh
    xx_test, yy_test = np.meshgrid(np.arange(x_min_test, x_max_test, h_x_test), np.arange(y_min_test, y_max_test, h_y_test))
    x_test = np.c_[xx_test.ravel(), yy_test.ravel()]


    #######################
    # Calculate manifold samples
    x_min_data = X[:, 0].min() - 0.5
    x_max_data = X[:, 0].max() + 0.5
    int_range = (x_max_data - x_min_data)
    x0_samples = np.linspace(x_min_data+int_range*.15, x_max_data-int_range*.15, n_db_samples, endpoint = True).reshape(-1,1)
    x1_samples = f_db(x0_samples)
    
    # l2 distance weighting
    manifold_samples = np.hstack((x0_samples, x1_samples))
    dist = distances.l2_parallel(sim_points, manifold_samples)
    dist = utils_np.exp_kernel_func(dist, lam = rho, q = 2)
    dist = dist / dist.sum(axis = -1, keepdims = True) # normalize
    l2_weighting = np.outer(dist[0,:], dist[1,:])

    # Exponential Geodesic Kernel
    geo_distance = distances.geodistance_toy_1d(f_db)
    geo_matrix = distances.geomatrix(x0_samples, geo_distance)
    geo_weighting = utils_np.exp_kernel_func(geo_matrix, lam = lam, q = 1)

    # WEG Kernel
    sim_points_WEG = np.multiply(geo_weighting, l2_weighting).sum()
    WEG_train = np.array([[1,sim_points_WEG],[sim_points_WEG,1]])
    noise = exp_var * np.identity(2)

    WEG_test = WEG(sim_points, x_test, rho, manifold_samples, geo_weighting)


    # tmp = WEG(x_test, x_test, rho, manifold_samples, geo_weighting)
    var_test = (np.identity(WEG_test.shape[1]) - WEG_test.transpose() @ np.linalg.inv(WEG_train + noise) @ WEG_test).diagonal()
    # import pdb; pdb.set_trace()

    [plot1, plot2, plot3] = figs.serve_prediction_plot(
        model=clf,
        X_train=X_train_tf,
        X_test=X_test_tf,
        y_train=y_train,
        y_test=y_test,
        Z=Z,
        xx=xx,
        yy=yy,
        mesh_step=[h_x, h_y],
        threshold=threshold,
        db_samples=[x0_samples, x1_samples],
        db_domain=[X[:, 0].min() - 4, X[:, 0].max() + 4],
        data_transform = data_transform,
        plot_lim = [x_min,x_max, y_min,y_max],
        f_db = f_db,
        sim_points = sim_points,
        sim_points_WEG = sim_points_WEG,
        l2_weighting = dist,
        xx_uncertainty = xx_test,
        yy_uncertainty = yy_test,
        step_uncertainty = [h_x_test, h_y_test],
        uncertainty_values = var_test,
    )

    #===========================================
    # HEATMAP 1
    #===========================================
    zmax = geo_matrix.max()*1.3
    zmin = 0
    # cmap = colormaps['BuPu']
    # colorscale1 = utils_io.shiftedColorMap(cmap, start = 0, midpoint = zmax / 2, stop = zmax)
    colorscale1 = 'PuBu'
    # colorscale1 = [[0, "#FFFFFF"], [1, "#3674AE"]]

    
    #===========================================
    # HEATMAP 2
    #===========================================
    if comparison == 'linear' or comparison == 'linear_ratio':
        poly_power2 = 1
        colorscale2 = colorscale1
        zmax2 = zmax
        zmin2 = zmin
        # labelcolor = 'rgba(255,255,255,0.5)'
        labelcolor = 'rgba(75,75,75,0.5)'
        title2 = "Geodesic Distance Matrix: K = %s (Linear)" % str(poly_power2)

        # Train Model
        data_transform = poly_transform_x0(power = poly_power2)
        X_train_tf = data_transform(X_train)
        clf = LogisticRegression(penalty = 'none')
        clf.fit(X_train_tf, y_train)
        f_db = f_db_linear(clf, data_transform)

        # int_range = (x_max_data - x_min_data)
        # x0_samples = np.linspace(x_min_data+int_range*.25, x_max_data-int_range*.25, n_db_samples, endpoint = True).reshape(-1,1)
        # x1_samples = f_db(x0_samples)

        # Calculate Geodesic Matrix
        geo_distance = distances.geodistance_toy_1d(f_db)
        geo_matrix2 = distances.geomatrix(x0_samples, geo_distance)

        if comparison == 'linear_ratio':
            # import pdb; pdb.set_trace()
            geo_matrix2 = np.divide(geo_matrix, geo_matrix2, out=np.zeros_like(geo_matrix2), where=geo_matrix2!=0)

            cmap = colormaps['RdBu_r']
            zmax2 = max(np.abs(geo_matrix2).max()*1.3,2)
            colorscale2 = utils_io.shiftedColorMap(cmap, start = 0, midpoint = 1, stop = zmax2)
            zmin2 = 0
            labelcolor = 'rgba(75,75,75,0.5)'
            title2 = "Ratio of Distances: K = %s, K = %s" % (str(poly_power), str(1))

    elif comparison == 'euclidean' or comparison == 'euclidean_ratio':
        colorscale2 = colorscale1
        zmax2 = zmax
        zmin2 = zmin
        # labelcolor = 'rgba(255,255,255,0.5)'
        labelcolor = 'rgba(75,75,75,0.5)'
        title2 = "Euclidean Distance Matrix"

        # Calculate Geodesic Matrix
        geo_distance = distances.eucdistance_toy_1d(f_db)
        geo_matrix2 = distances.geomatrix(x0_samples, geo_distance)

        if comparison == 'euclidean_ratio':
            geo_matrix2 = np.divide(geo_matrix, geo_matrix2, out=np.zeros_like(geo_matrix2), where=geo_matrix2!=0)

            cmap = colormaps['RdBu_r']
            zmax2 = max(np.abs(geo_matrix2).max()*1.3,2)
            colorscale2 = utils_io.shiftedColorMap(cmap, start = 0, midpoint = 1, stop = zmax2)
            zmin2 = 0
            labelcolor = 'rgba(75,75,75,0.5)'
            title2 = "Ratio of Geodesic and Eulidean Distances"

    heatmap1 = figs.serve_heatmap(
        matrix = geo_matrix,
        colorscale = colorscale1,
        zmin = zmin,
        zmax = zmax,
        title = "Geodesic Distance Matrix: K = %s" % poly_power,
        labelcolor = 'rgba(75,75,75,0.5)',
    )
    heatmap2 = figs.serve_heatmap(
        matrix = geo_matrix2,
        colorscale = colorscale2,
        zmin = zmin2,
        zmax = zmax2,
        title = title2,
        labelcolor = labelcolor,
    )

    #===========================================
    # HEATMAP 3
    #===========================================
    # import pdb; pdb.set_trace()
    heatmap3 = figs.serve_heatmap(
        matrix = geo_weighting,
        colorscale = "PuBu",
        zmin = 0,
        zmax = geo_weighting.max()*1.3,
        title = "EG Kernel Matrix: &#955; = %s" % str(np.round(lam,2)),
        labelcolor = 'rgba(75,75,75,0.5)',
        rounding = 2,
        fill_diagonal = False,
    )
    #===========================================
    # HEATMAP 4
    #===========================================
    # import pdb; pdb.set_trace()
    heatmap4 = figs.serve_heatmap(
        matrix = l2_weighting,
        colorscale = "PuBu",
        zmin = 0,
        zmax = l2_weighting.max()*1.3,
        title = "Weighting Matrix: &#961; = %s" % str(np.round(rho,2)),
        labelcolor = 'rgba(75,75,75,0.5)',
        rounding = 2,
        fill_diagonal = False,
    )



    return plot1, heatmap1, heatmap2, plot2, heatmap3, heatmap4, plot3


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port = 8040)
