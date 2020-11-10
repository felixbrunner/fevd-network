import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

import networkx as nx
import matplotlib.pyplot as plt



#import scipy as sp

#import missingno as mno


import src


def run_estimation(year, var_grid, cov_grid, horizon):
    #data
    preprocessed_data = load_preprocess(year)
    df_volas = src.loader.load_year(year, data='volas')
    df_spy_vola = src.loader.load_spy_vola(year)
    tickers = src.loader.load_year_tickers(year)
    column_to_ticker = dict(pd.read_csv('../data/processed/annual/{}/tickers.csv'.format(year))['permno_to_ticker'])
    
    # estimate
    var_cv, var = estimate_var(preprocessed_data, var_grid)
    cov_cv, cov = estimate_cov(var.residuals_, cov_grid)
    
    # fevd
    fevd = make_fevd(var, cov, horizon)
    
    # descriptive
    var_desc = describe_var(var_cv, var)
    cov_desc = describe_cov(cov_cv, cov, var)
    fevd_desc = describe_fevd(fevd, horizon)
    desc = var_desc\
                .append(cov_desc)\
                .append(fevd_desc)
    
    # collect
    var_data = collect_var_data(var)
    cov_data = collect_cov_data(cov, var)
    fevd_data = collect_fevd_data(fevd, horizon, var)
    network_data = var_data\
                    .join(cov_data)\
                    .join(fevd_data)
    
    # save estimates
    desc.to_csv('../data/estimated/annual/{}/desc.csv'.format(year))
    network_data.to_csv('../data/estimated/annual/{}/network_data.csv'.format(year))
    
    # plots
    create_data_plots(year, preprocessed_data, df_volas, df_spy_vola) # data description
    create_var_plots(year, var_cv, var) # VAR
    create_cov_plots(year, cov_cv, cov) # Covariance matrix
    create_fevd_plots(year, fevd, horizon, column_to_ticker) # FEVD
    plt.close('all')

    return (preprocessed_data, var_cv, var, cov_cv, cov, fevd, desc, network_data)


def load_preprocess(year):
    data = src.loader.load_year(year, data='idio_vola')
    src.plot.missing_data(data,
                          save_path='../reports/figures/annual/{}/matrix_missing.pdf'.format(year))
    data = src.utils.log_replace(data, method='min')
    return data

def estimate_var(data, var_grid):
    var = src.VAR(data, p_lags=1)
    var_cv = var.fit_elastic_net_cv(grid=var_grid,
                                    return_cv=True)
    return (var_cv, var)

def estimate_cov(data, cov_grid):
    cov_cv = GridSearchCV(src.covar.AdaptiveThresholdEstimator(),
                          param_grid=cov_grid,
                          cv=12,
                          n_jobs=-1,
                          verbose=1)\
                    .fit(data)
    cov = cov_cv.best_estimator_
    return (cov_cv, cov)

def make_fevd(var, cov, horizon):
    fevd = src.FEVD(var.var_1_matrix_, cov.covar_)
    return fevd
    
def describe_var(var_cv, var):
    var_desc = pd.Series({'lambda': var_cv.best_params_['lambdau'],
                          'kappa': var_cv.best_params_['alpha'],
                          'VAR_matrix_density': (var.var_1_matrix_!=0).sum()/var.n_series**2,
                          'VAR_lost_df': (var.coef_!=0).sum(),
                          'mean_connection': var.var_1_matrix_.mean(),
                          'mean_abs_connection': abs(var.var_1_matrix_).mean(),
                          'VAR_asymmetry': src.utils.matrix_asymmetry(var.var_1_matrix_),
                          })
    return var_desc

def describe_cov(cov_cv, cov, var):
    var_desc = pd.Series({'Phi_delta': cov_cv.best_params_['confidence_level'],
                          'eta': cov_cv.best_params_['eta'],
                          'covar_density': (cov.covar_!=0).sum()/cov.covar_.shape[0]**2,
                          'covar_shrunk_by': 1-(cov.covar_/np.cov(var.residuals_, rowvar=False)).mean(),
                         })
    return var_desc

def describe_fevd(fevd, horizon):
    fevd_desc = pd.Series({'avg_connectedness': fevd.average_connectedness(horizon),
                           'fev_asymmetry': src.utils.matrix_asymmetry(fevd.decompose_fev(horizon, normalise=False)),
                           'fev_asymmetry_normalised': src.utils.matrix_asymmetry(fevd.decompose_fev(horizon, normalise=True)),
                           'fu_asymmetry': src.utils.matrix_asymmetry(fevd.decompose_fu(horizon, normalise=False)),
                           'fu_asymmetry_normalised': src.utils.matrix_asymmetry(fevd.decompose_fu(horizon, normalise=True)),
                          })
    return fevd_desc

def collect_var_data(var):
    var_data = pd.DataFrame(index=var.var_data.columns)
    var_data['VAR_intercept'] = var.intercepts_
    var_data['mean_abs_VAR_in'] = (abs(var.var_1_matrix_).sum(axis=1) - abs(np.diag(var.var_1_matrix_))) / (var.var_1_matrix_.shape[0]-1)
    var_data['mean_abs_VAR_out'] = (abs(var.var_1_matrix_).sum(axis=0) - abs(np.diag(var.var_1_matrix_))) / (var.var_1_matrix_.shape[0]-1)
    return var_data

def collect_cov_data(cov, var):
    cov_data = pd.DataFrame(index=var.var_data.columns)
    cov_data['residual_variance'] = np.diag(cov.covar_)
    cov_data['mean_resid_corr'] = (src.utils.cov_to_corr(cov.covar_).sum(axis=1)-1) / (cov.covar_.shape[0]-1)
    return cov_data

def collect_fevd_data(fevd, horizon, var):
    fevd_data = pd.DataFrame(index=var.var_data.columns)
    fevd_data['in_connectedness'] = fevd.in_connectedness(horizon=horizon)
    fevd_data['out_connectedness'] = fevd.out_connectedness(horizon=horizon)
    fevd_data['fev_others'] = fevd.fev_others(horizon=horizon)
    fevd_data['fev_all'] = fevd.fev_all(horizon=horizon)
    fevd_data['eigenvector_centrality'] = list(nx.eigenvector_centrality(fevd.to_fev_graph(horizon, normalise=False), weight='weight', max_iter=1000).values())
    fevd_data['closeness_centrality'] = list(nx.closeness_centrality(fevd.to_fev_graph(horizon, normalise=False), distance='weight').values())
    return fevd_data

def create_data_plots(year, preprocessed_data, df_volas, df_spy_vola):
    src.plot.corr_heatmap(preprocessed_data.corr(), title='Data Correlation',
                          save_path='../reports/figures/annual/{}/heatmap_data_correlation.pdf'.format(year))
    src.plot.corr_heatmap(src.utils.autocorrcoef(preprocessed_data, lag=1), title='Data Auto-Correlation (First order)',
                         save_path='../reports/figures/annual/{}/heatmap_data_autocorrelation.pdf'.format(year))
    src.plot.vola_timeseries(preprocessed_data, total_volas=df_volas, index_vola=df_spy_vola,
                          save_path='../reports/figures/annual/{}/line_timeseries.pdf'.format(year))
    src.plot.histogram(preprocessed_data.stack(),  title='Data distribution',
                       save_path='../reports/figures/annual/{}/histogram_data.pdf'.format(year))
    
def create_var_plots(year, var_cv, var):
    src.plot.net_cv_contour(var_cv, 15, logy=True,
                            save_path='../reports/figures/annual/{}/contour_VAR.pdf'.format(year))
    src.plot.corr_heatmap(var.var_1_matrix_, title='VAR(1) coefficient matrix',
                          vmin=-abs(var.var_1_matrix_).max(), vmax=abs(var.var_1_matrix_).max(),
                          save_path='../reports/figures/annual/{}/heatmap_VAR1_matrix.pdf'.format(year))
    src.plot.corr_heatmap(pd.DataFrame(var.residuals_).corr(), title='VAR Residual Correlation',
                          save_path='../reports/figures/annual/{}/heatmap_VAR_residual_correlation.pdf'.format(year))
    src.plot.corr_heatmap(src.utils.autocorrcoef(pd.DataFrame(var.residuals_), lag=1), title='VAR Residual Auto-Correlation (First order)',
                          save_path='../reports/figures/annual/{}/heatmap_VAR_residual_autocorrelation.pdf'.format(year))
    res_cov = pd.DataFrame(var.residuals_).cov()
    src.plot.corr_heatmap(res_cov, title='VAR Residual Sample Covariance Matrix', vmin=-abs(res_cov.values).max(), vmax=abs(res_cov.values).max(),
                          save_path='../reports/figures/annual/{}/heatmap_VAR_residual_covariance.pdf'.format(year))
    src.plot.histogram(var.residuals_.ravel(),  title='VAR Residual distribution',
                       save_path='../reports/figures/annual/{}/histogram_VAR_residuals.pdf'.format(year))
    
def create_cov_plots(year, cov_cv, cov):
    src.plot.cov_cv_contour(cov_cv, 15, logy=False,
                           save_path='../reports/figures/annual/{}/contour_cov.pdf'.format(year))
    src.plot.corr_heatmap(cov.covar_, title='Adaptive Threshold Estimate of VAR Residual Covariances',
                          vmin=-abs(cov.covar_).max(), vmax=abs(cov.covar_).max(),
                          save_path='../reports/figures/annual/{}/heatmap_cov_matrix.pdf'.format(year))
    
def create_fevd_plots(year, fevd, horizon, column_to_ticker):
    src.plot.corr_heatmap(pd.DataFrame(fevd.fev_single(horizon)), 'FEV Single Contributions',
                          vmin=-abs(fevd.fev_single(horizon)).max(), vmax=abs(fevd.fev_single(horizon)).max(),
                          save_path='../reports/figures/annual/{}/heatmap_FEVD_contributions.pdf'.format(year))
    src.plot.corr_heatmap(pd.DataFrame(fevd.decompose_fev(horizon=horizon, normalise=False))-np.diag(np.diag(fevd.decompose_fev(horizon=horizon, normalise=False))),
                          title='FEVD Decomposition (off-diagonal)', vmin=0, vmax=None, cmap='binary',
                          save_path='../reports/figures/annual/{}/heatmap_FEVD_decomposition.pdf'.format(year))
    src.plot.network_graph(fevd.to_fev_graph(horizon, normalise=False), column_to_ticker, title='FEVD connectedness (absolute)', red_percent=5, linewidth=0.25,
                           save_path='../reports/figures/annual/{}/network_absolute.png'.format(year))
    src.plot.network_graph(fevd.to_fev_graph(horizon, normalise=True), column_to_ticker, title='FEVD connectedness (%)', red_percent=2, linewidth=0.25,
                           save_path='../reports/figures/annual/{}/network_pct.png'.format(year))