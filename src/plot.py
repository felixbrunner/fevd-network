import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import missingno as mno

import src.utils

import sys
sys.path.append('../../kungfu/')
import kungfu as kf


def corr_heatmap(data, title='Correlation Matrix', vmin=-1, vmax=1, cmap='seismic', save_path=None, infer_limits=False, infer_vmax=False):
    '''Plots a numpy array or pandas dataframe as a heatmap.'''
    if type(data) == pd.DataFrame:
        data = data.values
        
    # limits
    if infer_limits:
        vmax = abs(data).max()
        vmin = -abs(data).max()
    elif infer_vmax:
        vmax = data.max()
        vmin = 0
    
    # create plot
    fig = plt.figure(figsize=(12, 10))
    plt.matshow(data, fignum=fig.number, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16)
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')

def net_cv_contour(cv, levels=12, logy=False, save_path=None):
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
    ax.legend()#loc='upper left')
    cb.set_label('Cross-validation MSE', rotation=90)
    if logy:
        ax.set_yscale('log')
    
    # limits
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([min(y_values), max(y_values)])

    # save
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
    

def cov_cv_contour(cv, levels=12, logy=False, save_path=None):
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
    ax.set_xlabel('$\Phi(\delta)$ (0.5=sample cov, 1=zeros)')#x_name)
    ax.set_ylabel('$\eta$ (0=zeros, 1=soft-thresholding, 2="ridge")')#y_name)
    ax.legend()#loc='upper left')
    cb.set_label('Cross-validation Loss', rotation=90)
    if logy:
        ax.set_yscale('log')
    
    # limits
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([min(y_values), max(y_values)])
    
    # save
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        

def get_edge_colors(graph, percentage=5, col1='grey', col2='firebrick'):
    imbalances = []
    for i,j in graph.edges():
        ij_weight = graph[i][j]['weight']
        ji_weight = graph[j][i]['weight']
        imbalances += [abs(ij_weight-ji_weight)]
    threshold = np.percentile(np.array(imbalances), 100-percentage)
    colors = [col2 if imb>threshold else col1 for imb in imbalances]
    return colors
    
    
def network_graph(graph, name_mapping=None, title=None, red_percent=0, save_path=None, linewidth=0.2, **kwargs):
    ''''''
    # relabel
    if name_mapping is not None:
        graph = nx.relabel_nodes(graph, name_mapping)
    
    # line weights
    weights = np.array([graph[i][j]['weight'] for i,j in graph.edges()])
    weights /= weights.mean()/linewidth
    
    # line colors
    colors = get_edge_colors(graph, percentage=red_percent)
    
    #plot
    fig, ax = plt.subplots(1, 1, figsize=[22,12])
    options = {'node_color': 'white',
               'edge_color': colors,
               'node_size': 0,
               'linewidths': 0,
               'with_labels': True,
               'width': weights,
               'font_weight': 'bold',
               'arrows': False,
              }
    nx.draw(graph, ax=ax, **options, **kwargs)
    ax.set_title(title)
    
    # save
    if save_path:
        fig.savefig(save_path, format='png', dpi=fig.dpi*2, bbox_inches='tight')


def missing_data(df, save_path=None):
    '''Creates and saves a missingno plot.'''
    fig = plt.figure()
    mno.matrix(df, labels=False)
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        
        
def vola_timeseries(idio_volas, total_volas=None, index_vola=None, save_path=None):
    ''''''
    fig, ax = plt.subplots(1, 1)
    ax.plot(idio_volas.mean(axis=1), label='Idiosyncratic variances: cross-sectional mean')
    if total_volas is not None:
        ax.plot(np.log(total_volas.mean(axis=1)), label='Total variances: cross-sectional mean')
    if index_vola is not None:
        ax.plot(np.log(index_vola), label='Index variance')
    kf.add_recession_bars(ax, startdate=idio_volas.index[0], enddate=idio_volas.index[-1])
    ax.legend()
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        
        
def histogram(df, bins=100, title='Data distribution', save_path=None):
    ''''''
    fig, ax = plt.subplots(1, 1)
    ax.hist(df, bins=bins)
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')