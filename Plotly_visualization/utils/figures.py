import colorlover as cl
import plotly.graph_objs as go
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
import itertools


def serve_prediction_plot(
    model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold, db_samples, db_domain, data_transform, plot_lim, f_db, sim_points, sim_points_WEG, l2_weighting, xx_uncertainty, yy_uncertainty, step_uncertainty, uncertainty_values,
):
    [h_x, h_y] = mesh_step
    [h_x_uncertainty, h_y_uncertainty] = step_uncertainty
    # Get train and test score from model
    y_pred_train = (model.decision_function(X_train) > threshold).astype(int)
    y_pred_test = (model.decision_function(X_test) > threshold).astype(int)
    train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    [x_min, x_max, y_min, y_max] = plot_lim

    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    range = max(abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max()))

    # Colorscale
    bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    # bright_cscale = [[0, "#e3ba02"], [1, "#0b8bff"]]
    cscale = [
        [0.0000000, "#ff744c"],
        [0.1428571, "#ff916d"],
        [0.2857143, "#ffc0a8"],
        [0.4285714, "#ffe7dc"],
        [0.5714286, "#e5fcff"],
        [0.7142857, "#c8feff"],
        [0.8571429, "#9af8ff"],
        [1.0000000, "#20e6ff"],
    ]
    # cscale = [
    #     # [0.0000000, "#fce26d"],
    #     [0.0000000, "#fae89b"],
    #     [1.0000000, "#20e6ff"],
    # ]

    data = []
    data2 = []
    data3 = []

    # Plot the prediction contours
    fig = go.Contour(
        x=np.arange(xx.min(), xx.max(), h_x),
        y=np.arange(yy.min(), yy.max(), h_y),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - range,
        zmax=scaled_threshold + range,
        hoverinfo="none",
        showscale=False,
        contours=dict(showlines=False),
        colorscale=cscale,
        opacity=0.8,
        # line_smoothing = 1,
    )
    data.append(fig)

    # Plot uncertainty contours
    fig = go.Contour(
        x=np.arange(xx_uncertainty.min(), xx_uncertainty.max(), h_x_uncertainty),
        y=np.arange(yy_uncertainty.min(), yy_uncertainty.max(), h_y_uncertainty),
        z=uncertainty_values,
        # zmin=scaled_threshold - range,
        # zmax=scaled_threshold + range,
        # hoverinfo="none",
        # showscale=False,
        contours=dict(showlines=False),
        colorscale='Deep',
        opacity=0.8,
        # line_smoothing = 1,
    )
    data3.append(fig)



    # Plot the Decision Boundary
    x_tmp = np.arange(db_domain[0],db_domain[1], 0.1)
    y_tmp = f_db(x_tmp).reshape(-1)

    fig = go.Scatter(
            x=x_tmp,
            y = y_tmp,
            showlegend=False,
            mode = "lines",
            hoverinfo="none",
            line=dict(color="White", width =  12)
        )
    data.append(fig)
    data2.append(fig)
    data3.append(fig)

    fig = go.Scatter(
        x=x_tmp,
        y = y_tmp,
        mode = "lines",
        name = "Decision Boundary",
        hoverinfo="none",
        line=dict(color="Black", width =  6)
    )
    data.append(fig)
    data2.append(fig)
    data3.append(fig)


    # Plot Training Data
    fig = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode="markers",
        name=f"Training Data (accuracy={train_score:.3f})",
        marker=dict(size=13, symbol = 'triangle-up', color=y_train, colorscale=bright_cscale,
                line=dict(width=0.4,
                                        color='Black')),
    )
    data.append(fig)
    # data2.append(fig)


    # # Plot Test Data
    # data.append(go.Scatter(
    #     x=X_test[:, 0],
    #     y=X_test[:, 1],
    #     mode="markers",
    #     name=f"Test Data (accuracy={test_score:.3f})",
    #     marker=dict(
    #         size=10, symbol="triangle-up", color=y_test, colorscale=bright_cscale
    #     ),
    # ))
    # Plot DB Samples
    # coef = model.coef_
    # bias = model.intercept_
    # x0_samples = np.arange(x_min, x_max, (x_max - x_min) / n_db_samples).reshape(-1,1)
    # x1_samples = np.hstack((x0_samples, np.zeros_like(x0_samples)))
    # x1_samples = - (1/coef[0,1]) * (data_transform(x1_samples) @ coef.transpose() + bias)

    # Plot2: Plot projection lines
    [x0_samples, x1_samples] = db_samples
    line_iterator = itertools.product(np.arange(2), np.arange(x0_samples.shape[0]))
    for (i,j) in line_iterator:

        fig =go.Scatter(
            x = np.hstack((sim_points[i,0], x0_samples[j,0])),
            y = np.hstack((sim_points[i,1], x1_samples[j,0])),
            mode = "lines",
            line=dict(width=25 * l2_weighting[i,j], color='#282b38', dash = 'dot'),
            showlegend=False,
        )
        data2.append(fig)


    # Plot2: Plot Sim Points
    fig =go.Scatter(
        x = sim_points[:,0],
        y = sim_points[:,1],
        name = "Sample Explanations",
        mode = "markers+text",
        textposition="middle center",
        textfont=dict(
            family="sans serif",
            size=13,
            color="white",),
        text = ['<b>E1</b>', '<b>E2</b>'],
        marker=dict(
            size=30, symbol="star-diamond", color='#960363',
            line=dict(width=3, color='#282b38')),
    )
    data2.append(fig)
    data3.append(fig)


    # Plot DB Samples
    pal = sns.color_palette('deep', n_colors = x0_samples.shape[0])
    color_list = pal.as_hex()

    fig_list = []
    for k in np.arange(x0_samples.shape[0]):
        fig_list.append(go.Scatter(
            x=[x0_samples[k,0]],
            y=[x1_samples[k,0]],
            # mode="markers",
            name="Sample X%s" % str(k),
            mode="markers+text",
            textposition="middle center",
            text = '<b>X%s</b>' % str(k),
            marker=dict(
                size=35, symbol="square", color=color_list[k],
                line=dict(width=1,
                                        color='Black')),
            textfont=dict(
                family="sans serif",
                size=25,
                color="white",
            )))
    data.extend(fig_list)
    data2.extend(fig_list)
    data3.extend(fig_list)

    layout = go.Layout(
        xaxis=dict(ticks="", showticklabels=False, showgrid=True, zeroline=False, range=[x_min, x_max]),
        yaxis=dict(ticks="", showticklabels=False, showgrid=True, zeroline=False, range=[y_min, y_max]),
        hovermode="closest",
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        # plot_bgcolor="white",
        font={"color": "#a5b1cd"},
        # autosize=True,
    )

    layout2 = go.Layout(
        xaxis=dict(ticks="", showticklabels=False, showgrid=True, zeroline=False, range=[x_min, x_max]),
        yaxis=dict(ticks="", showticklabels=False, showgrid=True, zeroline=False, range=[y_min, y_max]),
        hovermode="closest",
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(255,255,255,0.55)",
        # plot_bgcolor="#A0A1A1",
        paper_bgcolor="#282b38",
        # plot_bgcolor="white",
        font={"color": "#a5b1cd"},
        # autosize=True,
        annotations=[
            go.layout.Annotation(
                text="<b>K<sub>WEG</sup>(E<sub>1</sup>, E<sub>2</sup>) = %s </b>" % str(np.round(sim_points_WEG,3)),
                font={"color": "Black", 'size': 18},
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0,
                y=0,
                # bordercolor='black',
                # borderwidth=1
            )
        ]
    )

    return [
        go.Figure(data=data, layout=layout),
        go.Figure(data=data2, layout=layout2),
        go.Figure(data=data3, layout=layout),
    ]



def serve_heatmap(matrix, colorscale, title, zmin, zmax, labelcolor = 'rgba(255,255,255,0.5)', rounding = 1, fill_diagonal = True):

    data = []
    if matrix.shape[0]>15:
        largematrix = True
    else:
        largematrix = False

    labels = ["X%s"%str(i+1) for i in range(matrix.shape[0])]
    labels = [""]+labels
    tmp_matrix = np.zeros((matrix.shape[0]+1, matrix.shape[1]+1))
    tmp_matrix[1:,1:] = matrix

    if largematrix:
        label_matrix = None
    else:
        label_matrix = tmp_matrix.round(rounding).astype('str')
        if fill_diagonal: np.fill_diagonal(label_matrix, '')
    data.append(go.Heatmap(
        z = tmp_matrix.tolist(),
        x = labels,
        y = labels,
        text =label_matrix,
        texttemplate="%{text}",
        textfont={"size":10,'color':labelcolor},
        name = 'Geodesic Distance',
        colorscale=colorscale,
        zmin = zmin,
        zmax = zmax,
        # zauto=False,
        # hoverinfo='text',
        # text = tmp_matrix,
        ))

    # Border Background
    fixed_scale = [[0, 'rgba(1,1,1,0)'], [1, 'black']]
    tmp_matrix = np.zeros((matrix.shape[0]+1, matrix.shape[1]+1))
    tmp_matrix[0,:] = 1
    tmp_matrix[:,0] = 1
    data.append(go.Heatmap(
        z = tmp_matrix.tolist(),
        x = labels,
        y = labels,
        colorscale=fixed_scale,
        # hoverinfo='text',
        # text = tmp_matrix,
        hoverinfo="none",
        showscale = False,
        ))

    # Axis labels
    pal = sns.color_palette('deep', n_colors = matrix.shape[0])
    color_list = pal.as_hex()
    color_list = ['#282b38'] + color_list
    for i in range(matrix.shape[0]+1):
        tmp_matrix = np.zeros((matrix.shape[0]+1, matrix.shape[1]+1))
        if i == 0 and fill_diagonal:
            np.fill_diagonal(tmp_matrix, 1)
        else:
            tmp_matrix[0,i] = 1
            tmp_matrix[i,0] = 1

        xgap = ygap = 0

        if largematrix:
            label_matrix = np.empty(tmp_matrix.shape, dtype="<U8")
            if i != 0:
                label_matrix[0,i] = '%s' % str(i-1)
                label_matrix[i,0] = label_matrix[0,i]
                xgap = ygap = 1
        else:
            label_matrix = np.empty(tmp_matrix.shape, dtype="<U8")
            if i != 0:
                label_matrix[0,i] = 'X%s' % str(i-1)
                label_matrix[i,0] = label_matrix[0,i]
                xgap = ygap = 1

        if i <= 10:
            fontsize = 17
        else:
            fontsize = 12

        fixed_scale = [[0, 'rgba(1,1,1,0)'], [1, color_list[i]]]
        data.append(go.Heatmap(
            z = tmp_matrix.tolist(),
            # hover_data = False,
            showscale = False,
            text = label_matrix,
            colorscale = fixed_scale,
            texttemplate="%{text}",
            textfont={"size":fontsize,'color':'white','family':"sans serif",},
            hoverinfo="none",
            xgap = xgap,
            ygap = ygap,
            )
            )
    
    # hoverinfo

    labels = ["X%s"%str(i+1) for i in range(matrix.shape[0])]
    labels = [""]+labels
    tmp_matrix = np.zeros((matrix.shape[0]+1, matrix.shape[1]+1)).astype('str')
    tmp_matrix[1:,1:] = matrix.astype('str')
    tmp_matrix[0,:] = 'NaN'
    tmp_matrix[:,0] = 'NaN'

    # if largematrix:
    #     label_matrix = None
    # else:
    #     label_matrix = tmp_matrix.round(1).astype('str')
    #     np.fill_diagonal(label_matrix, '')
    fixed_scale = [[0, 'rgba(1,1,1,0)'], [1, 'rgba(1,1,1,0)']]
    data.append(go.Heatmap(
        z = tmp_matrix.tolist(),
        x = labels,
        y = labels,
        # text =tmp_matrix.tolist(),
        # texttemplate="%{text}",
        # textfont={"size":10,'color':labelcolor},
        name = 'Geodesic Distance',
        colorscale=fixed_scale,
        # zauto=False,
        hoverinfo='z',
        showscale = False
        ))

    layout = go.Layout(
        title=title,
        titlefont=dict(size = 25, family = 'sans serif'),
        margin=dict(l=50, r=50, t=100, b=10),
        legend=dict(bgcolor="#282b38", font={"color": "#a5b1cd"}, orientation="h"),
        yaxis=dict(autorange = "reversed", visible=False),
        xaxis=dict(visible=False),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    figure = go.Figure(data=data, layout=layout)

    return figure
