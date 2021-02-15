import numpy as np
import scipy as sp
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

def net_cv_contour(cv, levels=12, logx=False, logy=False, save_path=None):
    '''Creates a countour plot from cross-validation
    for hyperparamter search.
    '''
    # create plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    ax.set_title('Adaptive Elastic Net Hyper-Parameter Search Grid')

    # data
    x_name, y_name =  cv.param_grid.keys()
    x_values, y_values =  cv.param_grid.values()
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_values = -cv.cv_results_['mean_test_score'].reshape(len(x_values), len(y_values)).T

    # contour plotting
    contour = ax.contourf(x_grid, y_grid, z_values, levels=levels, cmap='RdYlGn_r', antialiased=True, alpha=1)
    ax.contour(x_grid, y_grid, z_values, levels=levels, colors='k', antialiased=True, linewidths=1, alpha=0.6)
    ax.contour(x_grid, y_grid, z_values, levels=[1.0], colors='k', antialiased=True, linewidths=2, alpha=1)
    cb = fig.colorbar(contour)

    # grid & best estimator
    x_v = [a[x_name] for a in cv.cv_results_['params']]
    y_v = [a[y_name] for a in cv.cv_results_['params']]
    ax.scatter(x_v, y_v, marker='.', label='grid', color='k', alpha=0.25)
    ax.scatter(*cv.best_params_.values(), label='best estimator', marker='x', s=150, color='k', zorder=2)

    # labels & legend
    ax.set_xlabel('$\kappa$ (0=ridge, 1=LASSO)')
    ax.set_ylabel('$\lambda$ (0=OLS, $\infty$=zeros)')
    ax.legend()#loc='upper left')
    cb.set_label('Cross-Validation MSE (Standardized data)', rotation=90)
    v = (1-cb.vmin) / (cb.vmax-cb.vmin)
    cb.ax.plot([0, 1], [v, v], 'k', linewidth=2)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    # limits
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([min(y_values), max(y_values)])

    # save
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        
def net_scatter_losses(cv, save_path=None):
    # extract data
    train_losses = -cv.cv_results_['mean_train_score']
    valid_losses = -cv.cv_results_['mean_test_score']
    lambdas = pd.Series([d['lambdau'] for d in cv.cv_results_['params']])
    kappas = pd.Series([d['alpha'] for d in cv.cv_results_['params']])
    best = cv.best_index_
    
    # figure parameters
    fig, ax = plt.subplots(1, 1)
    colors = np.log(lambdas)
    sizes = ((np.log(kappas)+12)*20)
    
    # labels
    ax.set_xlabel('Mean Training MSE (In-sample)')
    ax.set_ylabel('Mean Validation MSE (Out-of-sample)')
    ax.set_title('Adaptive Elastic Net Cross-Validation Errors')
    
    # scatter plots
    sc = ax.scatter(train_losses, valid_losses,
                         c=colors, s=sizes,
                         cmap='bone', edgecolor='k')
    ax.scatter(train_losses[best],
               valid_losses[best],
               s=sizes[best]*2, c='r', edgecolor='k', marker='x',
               zorder=100, label='best model')
    
    # 45 degree line
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, color='grey', linestyle='--', label='45-degree line', zorder=0)

    # legends
    handles, _ = sc.legend_elements(prop="colors", num=colors.nunique())
    color_legend = ax.legend(handles[2:], ['{:.1e}'.format(i) for i in lambdas.unique()], loc="lower left", title='λ')
    ax.add_artist(color_legend)
    
    handles, _ = sc.legend_elements(prop="sizes", alpha=0.6, num=sizes.nunique())
    size_legend = ax.legend(handles, ['{:.1e}'.format(i) for i in kappas.unique()], loc="lower right", title='κ')
    ax.add_artist(size_legend)
    ax.legend(loc='lower center')

    # save
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
    

def cov_cv_contour(cv, levels=12, logx=False, logy=False, save_path=None):
    '''Creates a countour plot from cross-validation
    for hyperparamter search.
    '''
    # create plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    ax.set_title('Adaptive Threshold Estimation Hyper-Parameter Search Grid')

    # data
    x_name, y_name =  cv.param_grid.keys()
    x_values, y_values =  cv.param_grid.values()
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_values = -cv.cv_results_['mean_test_score'].reshape(len(x_values), len(y_values)).T

    # contour plotting
    contour = ax.contourf(x_grid, y_grid, z_values, levels=levels, cmap='RdYlGn_r', antialiased=True, alpha=1)
    ax.contour(x_grid, y_grid, z_values, levels=levels, colors='k', antialiased=True, linewidths=1, alpha=0.6)
    cb = fig.colorbar(contour)

    # grid & best estimator
    x_v = [a[x_name] for a in cv.cv_results_['params']]
    y_v = [a[y_name] for a in cv.cv_results_['params']]
    ax.scatter(x_v, y_v, marker='.', label='grid', color='k', alpha=0.25)
    ax.scatter(*cv.best_params_.values(), label='best estimator', marker='x', s=100, color='k')

    # labels & legend
    ax.set_xlabel('$\delta$ (0.5=sample cov, 1=zeros)')#x_name)
    ax.set_ylabel('$\eta$ (0=zeros, 1=soft-thresholding, 2="ridge")')#y_name)
    ax.legend()#loc='upper left')
    cb.set_label('Cross-Validation Loss', rotation=90)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    # limits
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([min(y_values), max(y_values)])
    
    # save
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        
def cov_scatter_losses(cv, save_path=None):
    # extract data
    train_losses = -cv.cv_results_['mean_train_score']
    valid_losses = -cv.cv_results_['mean_test_score']
    deltas = pd.Series([d['delta'] for d in cv.cv_results_['params']])
    etas = pd.Series([d['eta'] for d in cv.cv_results_['params']])
    best = cv.best_index_
    
    # figure parameters
    fig, ax = plt.subplots(1, 1)
    colors = deltas
    sizes = (etas*200)+50
    
    # labels
    ax.set_xlabel('Mean Training Loss (In-sample)')
    ax.set_ylabel('Mean Validation Loss (Out-of-sample)')
    ax.set_title('Adaptive Threshold Estimation Cross-Validation Errors')
    
    # scatter plots
    sc = ax.scatter(train_losses, valid_losses,
                         c=colors, s=sizes,
                         cmap='bone', edgecolor='k')
    ax.scatter(train_losses[best],
               valid_losses[best],
               s=sizes[best]*2, c='r', edgecolor='k', marker='x',
               zorder=100, label='best model')
    
    # 45 degree line
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, color='grey', linestyle='--', label='45-degree line', zorder=0)

    # legends
    color_legend = ax.legend(*sc.legend_elements(prop='colors', num=colors.nunique()), loc="upper left", title='δ')
    ax.add_artist(color_legend)
    handles, _ = sc.legend_elements(prop="sizes", alpha=0.6, num=sizes.nunique())
    size_legend = ax.legend(handles, [round(i, 2) for i in etas.unique()], loc="lower right", title='η')
    ax.add_artist(size_legend)
    ax.legend(loc='lower center')
    
    # save
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        
def plot_glasso_cv(cv, save_path=None):
    '''Plots glasso hyperparmater search through cross-validation.'''
    # create plot
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale('log')
    ax.set_xlabel('ρ (0=sample cov, $\infty = I_N \cdot diag(\hat{\Sigma}$))')
    ax.set_ylabel('Mean Cross-Validation Loss')
    ax.set_title('Graphical Lasso Hyper-Parameter Search Grid')
    
    # add elements
    ax.plot(cv.param_grid['alpha'], -cv.cv_results_['mean_test_score'], marker='o', label='mean validation loss')
    ax.plot(cv.param_grid['alpha'], -cv.cv_results_['mean_train_score'], marker='s', label='mean training loss', linestyle='--')
    # ax.axhline(-cov_cv.best_score_, label='Best Adaptive Threshold Estimate', linestyle=':', linewidth=1, color='k')
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.axvline(cv.best_params_['alpha'], label='best estimator', color='k', linestyle=':', linewidth=1)
    ax.scatter(cv.best_params_['alpha'], -cv.cv_results_['mean_test_score'][cv.best_index_], color='k', marker='o', zorder=100, s=100)
    ax.scatter(cv.best_params_['alpha'], -cv.cv_results_['mean_train_score'][cv.best_index_], color='k', marker='s', zorder=100, s=100) #colors[2]?
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        

def get_edge_colors(graph, percentage=5, col1='grey', col2='firebrick'):
    imbalances = []
    for i,j in graph.edges():
        try:
            ij_weight = graph[i][j]['weight']
            ji_weight = graph[j][i]['weight']
            imbalances += [abs(ij_weight-ji_weight)]
        except:
            pass
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
        fig.savefig(save_path, format='png', dpi=fig.dpi, bbox_inches='tight')


def missing_data(df, save_path=None):
    '''Creates and saves a missingno plot.'''
    fig = plt.figure()
    mno.matrix(df, labels=False)
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        
        
def var_timeseries(idio_var, total_var=None, index_var=None, save_path=None):
    ''''''
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    systematic = (total_var.mean(axis=1)-idio_var.mean(axis=1)).clip(0, None)
    
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Variance decomposition: cross-sectional means')
    
    ax.plot(total_var.mean(axis=1), label='Total variances', c=colors[0])
    
    ax.plot(systematic, c=colors[1], linewidth=2, linestyle='-')
    ax.fill_between(idio_var.index, 0, systematic, alpha=0.5, label='Systematic variance contribution', color=colors[1])
    
    ax.fill_between(idio_var.index, systematic, total_var.mean(axis=1), alpha=0.5, label='Non-systematic variance contribution', color=colors[0])
    
    ax.plot(index_var, c=colors[2], label='SPY variance', linestyle='--', alpha=0.6)
    
    ax.legend()
#     kf.add_recession_bars(ax, startdate=idio_volas.index[0], enddate=idio_volas.index[-1])
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
        
        
def histogram(df, bins=100, title='Data distribution', save_path=None, drop_tails=0):
    ''''''
    # drop outliers
    df_ = df.copy()
    df_ = df_[(df_.quantile(0+drop_tails/2)<df_.values) & (df_.values<df_.quantile(1-drop_tails/2))]
    
    # plot
    fig, ax = plt.subplots(1, 1)
    ax.hist(df_, bins=bins, label='Data')
    ax.set_title(title)
    kde = sp.stats.gaussian_kde(df_)
    xvals = np.linspace(*ax.get_xlim(), bins)
    ax.plot(xvals, kde(xvals)*(df_.max()-df_.min())/bins*len(df_)*(1-drop_tails), label='Scaled KDE', c='k')
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')