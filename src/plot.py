import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import src.utils


def corr_heatmap(data, title='Correlation'):
    '''Plots a numpy array or pandas dataframe as a heatmap.'''
    if type(data) == pd.DataFrame:
        data = data.values    
    f = plt.figure(figsize=(12, 10))
    plt.matshow(data, fignum=f.number, cmap='coolwarm')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16)
    
    

def cv_contour(cv, levels=12):
    '''Creates a countour plot from cross-validation
    for hyperparamter search.
    '''
    # create plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))

    # data
    x_name, y_name =  cv.param_grid.keys()
    x_values, y_values =  cv.param_grid.values()
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_values = -cv.cv_results_['mean_test_score'].reshape(len(x_values), len(y_values)).T

    # contour plotting
    contour = ax.contourf(x_grid, y_grid, z_values, levels=levels, cmap='RdYlGn_r', antialiased=True, alpha=1)
    ax.contour(x_grid, y_grid, z_values, levels=levels, colors='k', antialiased=True, linewidths=1)
    cb = fig.colorbar(contour)

    # grid & best estimator
    x_v = [a[x_name] for a in cv.cv_results_['params']]
    y_v = [a[y_name] for a in cv.cv_results_['params']]
    ax.scatter(x_v, y_v, marker='.', label='grid', color='k', alpha=0.25)
    ax.scatter(*cv.best_params_.values(), label='best estimator', marker='x', s=100, color='k')

    # labels & legend
    ax.set_xlabel('$\kappa$ (0=ridge, 1=LASSO)')#x_name)
    ax.set_ylabel('$\lambda$ (0=OLS)')#y_name)
    ax.legend(loc='upper left')
    cb.set_label('Cross-validation MSE', rotation=90)
    
    # limits
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([min(y_values), max(y_values)])
    
    
def network_graph(graph, name_mapping=None):
    ''''''
    # relabel
    if name_mapping is not None:
        graph = nx.relabel_nodes(graph, name_mapping)
    
    #plot
    options = {"node_color": "blue",
           'edge_color': 'grey',
           "node_size": 0,
           "linewidths": 0,
           "width": 0.1,
           'with_labels': True
          }
    nx.draw(graph, **options)