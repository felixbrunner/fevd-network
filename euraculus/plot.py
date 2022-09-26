import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as mno
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.decomposition import PCA

import kungfu as kf
from kungfu.plotting import add_recession_bars

# %% Plot settings

# style
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = [17, 8]

# colors
DARK_BLUE = "#014c63"
DARK_GREEN = "#3b5828"
DARK_RED = "#880000"
ORANGE = "#ae3e00"
DARK_YELLOW = "#ba9600"
PURPLE = "#663a82"
color_list = [DARK_BLUE, DARK_GREEN, DARK_RED, DARK_YELLOW, ORANGE, PURPLE]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_list)


def corr_heatmap(
    data: np.ndarray,
    title: str = "Correlation Matrix",
    labels: list = None,
    secondary_labels: list = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "seismic",
    save_path: str = None,
    infer_limits: bool = False,
    infer_vmax: bool = False,
):
    """Plots a numpy array or pandas dataframe as a heatmap.

    Args:
        data: Square data matrix.
        title: Title string.
        labels: List of labels for the primary axes.
        secondary_labels: List of grouped labels for the secondary axes.
        vmin: Minimum value for the coloring of the heatmap, default=-1.
        vmax: Maximum value for the coloring of the heatmap, default=1.
        cmap: Colormap, default="seismic"
        save_path: Path to save the figure.
        infer_limits: Indicates if heatmap range is inferred, default=False.
        infer_vmax: Indicates if heatmap maximum is inferred, default=False.

    """
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    mat = ax.matshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(*reversed(np.where(data == 0)), marker="$0$", color="gray", s=5)
    ax.set_title(title, fontsize=16)

    # add colorbar
    cb = plt.colorbar(mat)
    cb.ax.tick_params(labelsize=14)

    # add primary labels
    if labels is not None:
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(data)))
        ax.set_yticklabels(labels, rotation=0, fontsize=7)
        ax.grid(False)

    # add secondary labels
    if secondary_labels is not None:
        lim = len(secondary_labels)
        group_labels = list(dict.fromkeys(secondary_labels))
        group_divisions = [secondary_labels.index(sec) for sec in group_labels] + [lim]
        text_locations = [
            (a + b) / 2 for a, b in zip(group_divisions[:-1], group_divisions[1:])
        ]

        for div in group_divisions[1:-1]:
            ax.axvline(div - 0.5, color="w", linewidth=1, linestyle="-")
            ax.axhline(div - 0.5, color="w", linewidth=1, linestyle="-")
            ax.axvline(div - 0.5, color="k", linewidth=1, linestyle=(0, (5, 5)))
            ax.axhline(div - 0.5, color="k", linewidth=1, linestyle=(0, (5, 5)))

        for label, location in zip(group_labels, text_locations):
            ax.text(
                location,
                lim,
                s=label,
                rotation=90,
                fontsize=7,
                rotation_mode="anchor",
                horizontalalignment="right",
                verticalalignment="center",
            )
            ax.text(
                lim,
                location,
                s=label,
                fontsize=7,
                horizontalalignment="left",
                verticalalignment="center",
            )
        ax.grid(False)

    # save
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def net_cv_contour(cv, levels=12, logx=False, logy=False, save_path=None):
    """Creates a countour plot from cross-validation
    for hyperparamter search.
    """
    # create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title("Adaptive Elastic Net Hyper-Parameter Search Grid")

    # data
    x_name, y_name = cv.param_grid.keys()
    x_values, y_values = cv.param_grid.values()
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_values = (
        -cv.cv_results_["mean_test_score"].reshape(len(x_values), len(y_values)).T
    )

    # contour plotting
    contour = ax.contourf(
        x_grid,
        y_grid,
        z_values,
        levels=levels,
        cmap="RdYlGn_r",
        antialiased=True,
        alpha=1,
    )
    ax.contour(
        x_grid,
        y_grid,
        z_values,
        levels=levels,
        colors="k",
        antialiased=True,
        linewidths=1,
        alpha=0.6,
    )
    ax.contour(
        x_grid,
        y_grid,
        z_values,
        levels=[1.0],
        colors="k",
        antialiased=True,
        linewidths=2,
        alpha=1,
    )
    cb = fig.colorbar(contour)

    # grid & best estimator
    x_v = [a[x_name] for a in cv.cv_results_["params"]]
    y_v = [a[y_name] for a in cv.cv_results_["params"]]
    ax.scatter(x_v, y_v, marker=".", label="grid", color="k", alpha=0.25)
    ax.scatter(
        *cv.best_params_.values(),
        label="best estimator",
        marker="x",
        s=150,
        color="k",
        zorder=2,
    )

    # labels & legend
    ax.set_xlabel("$\kappa$ (0=ridge, 1=LASSO)")
    ax.set_ylabel("$\lambda$ (0=OLS, $\infty$=zeros)")
    ax.legend()  # loc='upper left')
    cb.set_label("Cross-Validation MSE (Standardized data)", rotation=90)
    v = (1 - cb.vmin) / (cb.vmax - cb.vmin)
    cb.ax.plot([0, 1], [v, v], "k", linewidth=2)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    # limits
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([min(y_values), max(y_values)])

    # save
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def net_scatter_losses(cv, save_path=None):
    # extract data
    train_losses = -cv.cv_results_["mean_train_score"]
    valid_losses = -cv.cv_results_["mean_test_score"]
    lambdas = pd.Series([d["lambdau"] for d in cv.cv_results_["params"]])
    kappas = pd.Series([d["alpha"] for d in cv.cv_results_["params"]])
    best = cv.best_index_

    # figure parameters
    fig, ax = plt.subplots(1, 1)
    colors = np.log(lambdas)
    sizes = (np.log(kappas) + 12) * 20

    # labels
    ax.set_xlabel("Mean Training MSE (In-sample)")
    ax.set_ylabel("Mean Validation MSE (Out-of-sample)")
    ax.set_title("Adaptive Elastic Net Cross-Validation Errors")

    # scatter plots
    sc = ax.scatter(
        train_losses, valid_losses, c=colors, s=sizes, cmap="bone", edgecolor="k"
    )
    ax.scatter(
        train_losses[best],
        valid_losses[best],
        s=sizes[best] * 2,
        c="r",
        edgecolor="k",
        marker="x",
        zorder=100,
        label="best model",
    )

    # 45 degree line
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, color="grey", linestyle="--", label="45-degree line", zorder=0)

    # legends
    handles, _ = sc.legend_elements(prop="colors", num=colors.nunique())
    color_legend = ax.legend(
        handles[2:],
        ["{:.1e}".format(i) for i in lambdas.unique()],
        loc="lower left",
        title="λ",
    )
    ax.add_artist(color_legend)

    handles, _ = sc.legend_elements(prop="sizes", alpha=0.6, num=sizes.nunique())
    size_legend = ax.legend(
        handles,
        ["{:.1e}".format(i) for i in kappas.unique()],
        loc="lower right",
        title="κ",
    )
    ax.add_artist(size_legend)
    ax.legend(loc="lower center")

    # save
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def cov_cv_contour(cv, levels=12, logx=False, logy=False, save_path=None):
    """Creates a countour plot from cross-validation
    for hyperparamter search.
    """
    # create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title("Adaptive Threshold Estimation Hyper-Parameter Search Grid")

    # data
    x_name, y_name = cv.param_grid.keys()
    x_values, y_values = cv.param_grid.values()
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_values = (
        -cv.cv_results_["mean_test_score"].reshape(len(x_values), len(y_values)).T
    )

    # contour plotting
    contour = ax.contourf(
        x_grid,
        y_grid,
        z_values,
        levels=levels,
        cmap="RdYlGn_r",
        antialiased=True,
        alpha=1,
    )
    ax.contour(
        x_grid,
        y_grid,
        z_values,
        levels=levels,
        colors="k",
        antialiased=True,
        linewidths=1,
        alpha=0.6,
    )
    cb = fig.colorbar(contour)

    # grid & best estimator
    x_v = [a[x_name] for a in cv.cv_results_["params"]]
    y_v = [a[y_name] for a in cv.cv_results_["params"]]
    ax.scatter(x_v, y_v, marker=".", label="grid", color="k", alpha=0.25)
    ax.scatter(
        *cv.best_params_.values(), label="best estimator", marker="x", s=100, color="k"
    )

    # labels & legend
    ax.set_xlabel("$\delta$ (0.5=sample cov, 1=zeros)")  # x_name)
    ax.set_ylabel('$\eta$ (0=zeros, 1=soft-thresholding, 2="ridge")')  # y_name)
    ax.legend()  # loc='upper left')
    cb.set_label("Cross-Validation Loss", rotation=90)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    # limits
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([min(y_values), max(y_values)])

    # save
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def cov_scatter_losses(cv, save_path=None):
    # extract data
    train_losses = -cv.cv_results_["mean_train_score"]
    valid_losses = -cv.cv_results_["mean_test_score"]
    deltas = pd.Series([d["delta"] for d in cv.cv_results_["params"]])
    etas = pd.Series([d["eta"] for d in cv.cv_results_["params"]])
    best = cv.best_index_

    # figure parameters
    fig, ax = plt.subplots(1, 1)
    colors = deltas
    sizes = (etas * 200) + 50

    # labels
    ax.set_xlabel("Mean Training Loss (In-sample)")
    ax.set_ylabel("Mean Validation Loss (Out-of-sample)")
    ax.set_title("Adaptive Threshold Estimation Cross-Validation Errors")

    # scatter plots
    sc = ax.scatter(
        train_losses, valid_losses, c=colors, s=sizes, cmap="bone", edgecolor="k"
    )
    ax.scatter(
        train_losses[best],
        valid_losses[best],
        s=sizes[best] * 2,
        c="r",
        edgecolor="k",
        marker="x",
        zorder=100,
        label="best model",
    )

    # 45 degree line
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, color="grey", linestyle="--", label="45-degree line", zorder=0)

    # legends
    color_legend = ax.legend(
        *sc.legend_elements(prop="colors", num=colors.nunique()),
        loc="upper left",
        title="δ",
    )
    ax.add_artist(color_legend)
    handles, _ = sc.legend_elements(prop="sizes", alpha=0.6, num=sizes.nunique())
    size_legend = ax.legend(
        handles, [round(i, 2) for i in etas.unique()], loc="lower right", title="η"
    )
    ax.add_artist(size_legend)
    ax.legend(loc="lower center")

    # save
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_glasso_cv(cv, save_path=None):
    """Plots glasso hyperparmater search through cross-validation."""
    # create plot
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log")
    ax.set_xlabel("ρ (0=sample cov, $\infty = diag(\hat{\Sigma}$))")
    ax.set_ylabel("Mean Cross-Validation Loss")
    ax.set_title("Graphical Lasso Hyper-Parameter Search Grid")

    # add elements
    ax.plot(
        cv.param_grid["alpha"],
        -cv.cv_results_["mean_test_score"],
        marker="o",
        label="mean validation loss",
    )
    ax.plot(
        cv.param_grid["alpha"],
        -cv.cv_results_["mean_train_score"],
        marker="s",
        label="mean training loss",
        linestyle="--",
    )
    # ax.axhline(-cov_cv.best_score_, label='Best Adaptive Threshold Estimate', linestyle=':', linewidth=1, color='k')

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.axvline(
        cv.best_params_["alpha"],
        label="best estimator",
        color="k",
        linestyle=":",
        linewidth=1,
    )
    ax.scatter(
        cv.best_params_["alpha"],
        -cv.cv_results_["mean_test_score"][cv.best_index_],
        color="k",
        marker="o",
        zorder=100,
        s=100,
    )
    ax.scatter(
        cv.best_params_["alpha"],
        -cv.cv_results_["mean_train_score"][cv.best_index_],
        color="k",
        marker="s",
        zorder=100,
        s=100,
    )  # colors[2]?
    ax.legend()

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def get_edge_colors(graph, percentage=5, col1="grey", col2="firebrick"):
    imbalances = []
    for i, j in graph.edges():
        try:
            ij_weight = graph[i][j]["weight"]
            ji_weight = graph[j][i]["weight"]
            imbalances += [abs(ij_weight - ji_weight)]
        except:
            pass
    threshold = np.percentile(np.array(imbalances), 100 - percentage)
    colors = [col2 if imb > threshold else col1 for imb in imbalances]
    return colors


def draw_fevd_as_network(
    fevd,
    df_info,
    horizon: int,
    table_name: str = "fevd",
    normalize: bool = False,
    title: str = "Network",
    save_path: str = None,
    pos: dict = None,
    **kwargs,
) -> dict:
    """Draw FEVD as a force layout network plot.

    Args:
        fevd: Forecast error variance decomposition object.
        df_info: Accompanying information in a dataframe.
        horizon: Horizon to construct connectedness table.
        table_name: Name of the table to be plotted.
        normalize: Indicates if the table shall be row-normalized.
        title: Title of the plot, default="Network".
        save_path: Path to save the figure.
        pos: Dictionary with initial locations for all/some nodes.

    Returns:
        layout: A dictionary of positions keyed by node.

    """
    # set up graph
    ticker_dict = {i: tick for (i, tick) in enumerate(df_info["ticker"].values)}
    g = fevd.to_graph(horizon=horizon, table_name=table_name, normalize=normalize)
    g = nx.relabel_nodes(g, ticker_dict)
    table = fevd._get_table(name="fev", horizon=horizon, normalize=normalize)

    # set layout
    if pos is None:
        layout = nx.fruchterman_reingold_layout(
            G=g, k=None, iterations=1000, dim=2, seed=0
        )
    else:
        # add new nodes to old graph, then fix 10% of nodes to stop rotation
        fixed = [p for p in pos if (p in g.nodes())]
        pos = nx.fruchterman_reingold_layout(
            G=g, k=None, iterations=500, dim=2, seed=0, pos=pos, fixed=fixed
        )
        fixed = np.random.choice(list(pos), len(pos) // 10)
        pos = nx.fruchterman_reingold_layout(
            G=g, k=None, iterations=500, dim=2, seed=0, pos=pos, fixed=fixed
        )
        layout = nx.fruchterman_reingold_layout(
            G=g, k=None, iterations=500, dim=2, seed=0, pos=pos
        )

    # set up node colors
    sector_colors = plt.get_cmap("Paired")(np.linspace(0, 1, 12))
    ff_sector_tickers = [
        "NoDur",
        "Durbl",
        "Manuf",
        "Enrgy",
        "Chems",
        "BusEq",
        "Telcm",
        "Utils",
        "Shops",
        "Hlth",
        "Money",
        "Other",
    ]
    ff_sector_codes = {tick: i for i, tick in enumerate(ff_sector_tickers)}

    # calculations to highlight nodes/edges
    out_eigenvector_centrality = fevd.out_eigenvector_centrality(
        horizon=horizon,
        table_name=table_name,
    )
    out_page_rank = fevd.out_page_rank(
        horizon=horizon,
        table_name=table_name,
        weights=df_info["mean_valuation_volatility"].values.reshape(-1, 1),
    )
    include_edges = table > np.percentile(table, q=90, axis=None, keepdims=True)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=[22, 12])
    ax.set_title(title)
    ax.grid(False)

    # draw nodes
    node_options = {
        "node_size": 750
        + (
            df_info["mean_valuation_volatility"]
            / df_info["mean_valuation_volatility"].mean()
        ).values
        * 250,
        "node_color": [
            sector_colors[ff_sector_codes[i]]
            for i in df_info["ff_sector_ticker"].values
        ],
        "linewidths": [
            4
            if e > np.percentile(out_eigenvector_centrality, 80)
            else 4
            if p > np.percentile(out_page_rank, 80)
            else 0
            for (e, p) in zip(out_eigenvector_centrality, out_page_rank)
        ],
        "alpha": 0.9,
        "edgecolors": [
            "w"
            if e > np.percentile(out_eigenvector_centrality, 80)
            else "grey"
            if p > np.percentile(out_page_rank, 80)
            else "r"
            for (e, p) in zip(out_eigenvector_centrality, out_page_rank)
        ],
        "node_shape": "o",
    }
    nx.draw_networkx_nodes(G=g, pos=layout, ax=ax, **node_options)

    # label nodes
    label_options = {
        # "labels": {i: tick for i, tick in enumerate(df_info["ticker"].values)},
        "font_weight": "bold",
        "font_size": 8,
    }
    nx.draw_networkx_labels(G=g, pos=layout, ax=ax, **label_options)

    # draw edges
    edge_options = {
        "arrows": True,  # False,#True,
        "connectionstyle": "arc3,rad=0.2",
        "node_size": node_options["node_size"],
        "arrowsize": 10,
        # "edgelist": [(x, y) for x, y in zip(*np.where(include_edges.T))],
        "edgelist": [
            (ticker_dict[x], ticker_dict[y]) for x, y in zip(*np.where(include_edges.T))
        ],
        "width": (table.T.flatten() / (0.25 * table[include_edges.T].mean()))[
            include_edges.T.flatten()
        ],
        # "edge_color": "grey",
        # "edge_cmap": "binary",
    }
    nx.draw_networkx_edges(G=g, pos=layout, ax=ax, **edge_options)

    # legends
    sector_legend = ax.legend(
        handles=[
            mpl.patches.Patch(facecolor=sector_colors[i], label=ticker)
            for ticker, i in ff_sector_codes.items()
        ],
        title="Sectors",
        edgecolor="k",
        facecolor="lightgray",
    )
    influence_legend = ax.legend(
        handles=[
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markeredgecolor="w",
                label="Eigenvector centrality",
                markersize=10,
                linewidth=0,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markeredgecolor="grey",
                label="Page rank",
                markersize=10,
                linewidth=0,
            ),
        ],
        title="Most influential assets",
        loc="lower right",
        edgecolor="k",
    )
    ax.add_artist(sector_legend)
    ax.add_artist(influence_legend)

    # save
    if save_path:
        fig.savefig(save_path, format="png", dpi=fig.dpi, bbox_inches="tight")

    return layout


def missing_data(df, title: str = "Missing Data", save_path: str = None):
    """Creates and saves a missingno plot."""
    # plot
    ax = mno.matrix(df, labels=False)
    fig = plt.gcf()

    # label
    ax.set_title(title, fontsize=16)
    plt.xticks(np.arange(100), df.columns, rotation=90, fontsize=12)

    # save
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def var_timeseries(idio_var, total_var=None, index_var=None, save_path=None):
    """"""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    systematic = (total_var.mean(axis=1) - idio_var.mean(axis=1)).clip(0, None)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Variance decomposition: cross-sectional means")

    ax.plot(total_var.mean(axis=1), label="Total variances", c=colors[0])

    ax.plot(systematic, c=colors[1], linewidth=2, linestyle="-")
    ax.fill_between(
        idio_var.index,
        0,
        systematic,
        alpha=0.5,
        label="Systematic variance contribution",
        color=colors[1],
    )

    ax.fill_between(
        idio_var.index,
        systematic,
        total_var.mean(axis=1),
        alpha=0.5,
        label="Non-systematic variance contribution",
        color=colors[0],
    )

    ax.plot(index_var, c=colors[2], label="SPY variance", linestyle="--", alpha=0.6)

    ax.legend()
    #     add_recession_bars(ax, startdate=idio_volas.index[0], enddate=idio_volas.index[-1])
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def histogram(df, bins=100, title="Data distribution", save_path=None, drop_tails=0):
    """"""
    # drop outliers
    df_ = df.squeeze()
    df_ = df_[
        (df_.quantile(0 + drop_tails / 2) < df_)
        & (df_ < df_.quantile(1 - drop_tails / 2))
    ]

    # plot
    fig, ax = plt.subplots(1, 1)
    ax.hist(df_, bins=bins, label="Data")
    ax.set_title(title)
    kde = sp.stats.gaussian_kde(df_.squeeze())
    xvals = np.linspace(*ax.get_xlim(), bins)
    ax.plot(
        xvals,
        kde(xvals) * (df_.max() - df_.min()) / bins * len(df_) * (1 - drop_tails),
        label="Scaled KDE",
        c="k",
    )
    ax.axvline([df.mean()], color="grey", label="Sample mean", linestyle="--")
    # normal = sp.stats.norm

    ax.legend()

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_estimation_summary(df, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # hyperparameters
    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.set_title("Cross-validated hyperparameters")
    l1 = ax1.plot(
        df["lambda"],
        linestyle="-",
        label="λ, mean={}".format(df["lambda"].mean().round(2)),
        c=colors[0],
    )
    l2 = ax1.plot(
        df["rho"],
        linestyle="-.",
        label="ρ, mean={}".format(df["rho"].mean().round(2)),
        c=colors[1],
    )
    l3 = ax2.plot(
        df["kappa"],
        linestyle="--",
        label="κ, mean={:.1e}".format(df["kappa"].mean()),
        c=colors[2],
    )

    # l4 = ax2.plot(df['eta'], linestyle=':', label='η, mean={}'.format(df['eta'].mean().round(2)), c=colors[3])
    ax1.set_ylim([1e-3, 1e1])
    ax2.set_ylim([1e-5, 1e0])
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax2.grid(None)
    ax1.set_ylabel("Penalty hyperparameters (λ, ρ)", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax2.set_ylabel("L1 hyperparameter (κ)", color=colors[2])
    ax2.tick_params(axis="y", labelcolor=colors[2])
    lines = l1 + l2 + l3  # +l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(1.05, 0.5), loc="center left")
    add_recession_bars(ax1, freq="M", startdate=df.index[0], enddate=df.index[-1])

    # Losses
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.set_title("Cross-validation losses")
    ax1.set_ylim([0, 1])
    l11 = ax1.plot(
        df["var_cv_loss"],
        linestyle="-",
        label="VAR CV loss, mean={}".format(df["var_cv_loss"].mean().round(2)),
        c=colors[0],
    )
    l12 = ax1.plot(
        df["var_train_loss"],
        linestyle="--",
        label="VAR train loss, mean={}".format(df["var_train_loss"].mean().round(2)),
        c=colors[0],
    )
    ax1.set_ylabel("VAR MSE", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    # ax2.set_ylim([0, 500])
    ax2.grid(None)
    l21 = ax2.plot(
        df["covar_cv_loss"],
        linestyle="-.",
        label="Covariance CV loss, mean={}".format(df["covar_cv_loss"].mean().round(2)),
        c=colors[1],
    )
    l22 = ax2.plot(
        df["covar_train_loss"],
        linestyle=":",
        label="Covariance train loss, mean={}".format(
            df["covar_train_loss"].mean().round(2)
        ),
        c=colors[1],
    )
    ax2.set_ylabel("Covariance loss", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    lines = l11 + l12 + l21 + l22
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(1.05, 0.5), loc="center left")
    add_recession_bars(ax1, freq="M", startdate=df.index[0], enddate=df.index[-1])

    # R2
    ax1 = axes[2]
    ax2 = ax1.twinx()
    ax1.set_title("Goodness of fit")
    ax1.set_ylim([0, 1])
    l11 = ax1.plot(
        df["var_r2"],
        label="AEnet, mean={}".format(df["var_r2"].mean().round(2)),
        c=colors[0],
        linestyle="-",
    )
    l12 = ax1.plot(
        df["var_r2_ols"],
        label="OLS, mean={}".format(df["var_r2_ols"].mean().round(2)),
        c=colors[0],
        linestyle="--",
    )
    ax1.set_ylabel("VAR R²", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    ax2.grid(None)
    l21 = ax2.plot(
        df["cov_mean_likelihood"],
        label="GLASSO, mean={}".format(df["cov_mean_likelihood"].mean().round(2)),
        c=colors[1],
        linestyle="-.",
    )
    l22 = ax2.plot(
        df["cov_mean_likelihood_sample_estimate"],
        label="Sample covariance, mean={}".format(
            df["cov_mean_likelihood_sample_estimate"].mean().round(2)
        ),
        c=colors[1],
        linestyle=":",
    )
    ax2.set_ylabel("Covariance average log-likelihood", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    lines = l11 + l12 + l21 + l22
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(1.05, 0.5), loc="center left")
    add_recession_bars(ax1, freq="M", startdate=df.index[0], enddate=df.index[-1])

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_regularisation_summary(df, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    df = df.astype(float)

    # Degrees of Freedom
    ax = axes[0]
    ax.set_title("Degrees of freedom")
    ax.fill_between(
        df.index,
        0,
        df["var_df_used"],
        alpha=0.5,
        label="DFs used by VAR estimation, mean={}".format(
            int(df["var_df_used"].mean())
        ),
        color=colors[0],
    )
    ax.plot(df["var_df_used"], c=colors[0], linewidth=1)
    ax.fill_between(
        df.index,
        df["var_df_used"],
        df["var_df_used"] + df["cov_used_df"],
        alpha=0.5,
        label="DFs used by covariance estimation, mean={}".format(
            int(df["cov_used_df"].mean())
        ),
        color=colors[1],
    )
    ax.plot(df["var_df_used"] + df["cov_used_df"], c=colors[1], linewidth=1)
    ax.fill_between(
        df.index,
        df["var_df_used"] + df["cov_used_df"],
        df["nobs"],
        alpha=0.3,
        label="Remaining DFs, mean={}".format(
            int((df["nobs"] - df["var_df_used"] - df["cov_used_df"]).mean())
        ),
        color=colors[2],
    )
    ax.plot(
        df["nobs"],
        c=colors[2],
        label="Total data points, mean={}".format(int(df["nobs"].mean())),
    )
    ax.plot(
        df["var_regular_lost_df"],
        c=colors[0],
        label="Non-regularised VAR DFs ({})".format(
            int(df["var_regular_lost_df"].mean())
        ),
        linestyle="--",
        linewidth=1.5,
    )
    ax.plot(
        df["var_regular_lost_df"] + df["covar_regular_lost_df"],
        c=colors[1],
        label="Non-regularised total DFs ({})".format(
            int((df["var_regular_lost_df"] + df["covar_regular_lost_df"]).mean())
        ),
        linestyle="-.",
        linewidth=1.5,
    )
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

    # Sparsity
    ax = axes[1]
    ax.set_title("Estimate sparsity")
    ax.plot(
        1 - df["var_matrix_density"],
        linestyle="-",
        label="VAR matrix sparsity, mean={}".format(
            (1 - df["var_matrix_density"]).mean().round(2)
        ),
    )
    ax.plot(
        1 - df["precision_density"],
        linestyle="--",
        label="Precision matrix sparsity, mean={}".format(
            (1 - df["precision_density"]).mean().round(2)
        ),
    )
    ax.plot(
        1 - df["mean_density"],
        linestyle="-.",
        label="Overall estimate sparsity, mean={}".format(
            (1 - df["mean_density"]).mean().round(2)
        ),
    )
    ax.set_ylim([0, 1])
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

    # Shrinkage
    ax = axes[2]
    ax.set_title("Estimate shrinkage")
    ax.plot(
        df["var_nonzero_shrinkage"],
        linestyle="-",
        label="VAR matrix shrinkage, mean={}".format(
            (df["var_nonzero_shrinkage"]).mean().round(2)
        ),
    )
    ax.plot(
        df["precision_nonzero_shrinkage"],
        linestyle="--",
        label="Precision matrix shrinkage, mean={}".format(
            (df["precision_nonzero_shrinkage"]).mean().round(2)
        ),
    )
    # ax.plot(
    #     df["mean_shrinkage"],
    #     linestyle=":",
    #     label="Overall estimate shrinkage, mean={}".format(
    #         (df["mean_shrinkage"]).mean().round(2)
    #     ),
    # )
    ax.plot(
        df["covar_nonzero_shrinkage"],
        linestyle="-.",
        label="Covariance matrix shrinkage, mean={}".format(
            (df["covar_nonzero_shrinkage"]).mean().round(2)
        ),
    )
    ax.set_ylim([0, 1])
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_network_summary(df, save_path=None):
    # set up plot
    fig, axes = plt.subplots(1, 1, figsize=(20, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # connectedness
    ax1 = axes  # [0]
    l1 = ax1.plot(
        df["fev_avg_connectedness"],
        label="Average connectedness $c^{avg}$, mean="
        + str((df["fev_avg_connectedness"]).mean().round(2)),
        c=colors[0],
    )
    ax1.set_ylabel("Connectedness", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax1.set_ylim([0, 0.35])
    # ax1.set_yscale("log")

    # concentration
    ax2 = ax1.twinx()
    l2 = ax2.plot(
        df["fev_concentration_out_page_rank"].rolling(1).mean(),
        label="Network concentration, mean={}".format(
            (df["fev_concentration_out_page_rank"]).mean().round(2)
        ),
        linestyle="--",
        c=colors[1],
    )
    ax2.grid(None)
    ax2.set_ylabel("Concentration", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])
    ax2.set_ylim([0, 0.7])
    # ax3.set_yscale("log")

    # asymmetry
    ax3 = ax1.twinx()
    l3 = ax3.plot(
        df["fev_asymmetry"],
        label="Network directedness, mean={}".format(
            (df["fev_asymmetry"]).mean().round(2)
        ),
        linestyle="-.",
        c=colors[2],
    )
    ax3.grid(None)
    ax3.set_ylabel("Directedness", color=colors[2])
    ax3.tick_params(axis="y", labelcolor=colors[2])
    ax3.yaxis.set_label_coords(1.07, 0.5)
    ax3.tick_params(direction="out", pad=50)
    ax3.set_ylim([0, 0.35])
    # ax2.set_yscale("log")

    # asymmetry
    ax4 = ax1.twinx()
    l4 = ax4.plot(
        df["fev_amplification"],
        label="Network amplification, mean={}".format(
            (df["fev_amplification"]).mean().round(2)
        ),
        linestyle=(0, (5, 1)),
        c=colors[3],
    )
    ax4.grid(None)
    ax4.set_ylabel("Amplification", color=colors[3])
    ax4.tick_params(axis="y", labelcolor=colors[3])
    ax4.yaxis.set_label_coords(1.115, 0.5)
    ax4.tick_params(direction="out", pad=100)
    ax4.set_ylim([0.8, 1.15])
    # ax2.set_yscale("log")

    # figure formatting
    ax1.set_title("FEVD Network statistics")
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    add_recession_bars(ax1, freq="M", startdate=df.index[0], enddate=df.index[-1])

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_partial_r2(df, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Network stats
    ax.set_title("VAR Partial $R^2$")
    ax.plot(
        df["var_partial_r2_factors"],
        label="Partial $R^2$ factors, mean="
        + str((df["var_partial_r2_factors"]).mean().round(2)),
        c=colors[0],
    )
    ax.plot(
        df["var_partial_r2_var"],
        label="Partial $R^2$ spillovers, mean="
        + str((df["var_partial_r2_var"]).mean().round(2)),
        c=colors[1],
        linestyle="--",
    )
    ax.set_ylabel("Partial $R^2$")
    ax.legend()
    add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_ledoitwolf_test(df: pd.DataFrame, save_path=None):
    """"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    #
    ax1 = ax
    l1 = ax1.plot(
        df["innovation_diagonality_test_stat"],
        label="test statistic, mean="
        + str((df["innovation_diagonality_test_stat"]).mean().round(2)),
        c=colors[0],
    )
    ax1.set_ylabel("test statistic", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    #
    ax2 = ax1.twinx()
    l2 = ax2.plot(
        df["innovation_diagonality_p_value"],
        label="p-value, mean="
        + str((df["innovation_diagonality_test_stat"]).mean().round(2)),
        c=colors[1],
        linestyle="--",
    )
    ax2.grid(False)
    ax2.set_ylabel("p-value", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    #
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left")
    add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_pca_elbow(df: pd.DataFrame, n_pcs: int = 10, save_path=None):
    # create pca
    pca = PCA(n_components=n_pcs).fit(df)

    # plot data
    fig, ax = plt.subplots(1, 1)
    ax.plot(pca.explained_variance_ratio_, marker="o")

    # xticks
    ax.set_xticks(np.arange(n_pcs))
    ax.set_xticklabels(np.arange(n_pcs) + 1)

    # labels
    ax.set_title("Principal Components: Explained Variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_sampling_fractions(df_summary, save_path=None):

    # calculate data
    df_sampling = pd.DataFrame()
    df_sampling["n_assets"] = df_summary.groupby("sampling_date").size()
    df_sampling["total_mean_mcap"] = df_summary.groupby("sampling_date")[
        "mean_size"
    ].sum()
    df_sampling["100_mean_mcap"] = df_summary.groupby("sampling_date").apply(
        lambda x: x.sort_values("mean_valuation_volatility", ascending=False)[
            "mean_size"
        ]
        .iloc[:100]
        .sum()
    )
    df_sampling["total_mean_vv"] = df_summary.groupby("sampling_date")[
        "mean_valuation_volatility"
    ].sum()
    df_sampling["100_mean_vv"] = df_summary.groupby("sampling_date").apply(
        lambda x: x.sort_values("mean_valuation_volatility", ascending=False)[
            "mean_valuation_volatility"
        ]
        .iloc[:100]
        .sum()
    )

    # set up plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax2 = ax.twinx()
    ax.set_title("Proportion of our sample compared to the entire CRSP universe")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot lines
    l1 = ax.plot(
        df_sampling["100_mean_vv"] / df_sampling["total_mean_vv"],
        label="Valuation volatility",
    )
    l2 = ax.plot(
        df_sampling["100_mean_mcap"] / df_sampling["total_mean_mcap"],
        label="Market capitalization",
        linestyle="--",
    )
    l3 = ax2.plot(
        100 / df_sampling["n_assets"],
        label="Number of assets (right axis)",
        linestyle="-.",
        color=colors[2],
    )

    # format
    ax.set_ylim([0, 1])
    ax.set_yticks([i / 10 for i in range(11)])
    ax.set_yticklabels([f"{int(tick*100)}%" for tick in ax.get_yticks()])
    ax.set_ylabel("Valuation volatility & market capitalization")

    ax2.set_ylim([0, 0.1])
    ax2.set_yticks([i / 100 for i in range(11)])
    ax2.set_yticklabels([f"{int(tick*10)}%" for tick in ax.get_yticks()])
    ax2.grid(False)
    ax2.set_ylabel("Number of assets", color=colors[2])
    ax2.tick_params(axis="y", labelcolor=colors[2])

    add_recession_bars(
        ax, freq="M", startdate=df_sampling.index[0], enddate=df_sampling.index[-1]
    )

    # legend
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)  # , loc="center left")

    # save
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


def plot_mcap_concentration(df_summary, sampling_date, save_path):
    """"""
    # create plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax2 = ax.twinx()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.set_title(f"Firm Size Distribution ({sampling_date.date()})")

    # prepare data
    mcaps = df_summary["last_mcap"] * 1e3
    mcaps = mcaps.loc[mcaps > 0].sort_values(ascending=False).reset_index(drop=True)
    cumulative = pd.Series([0]).append(mcaps.cumsum()) / mcaps.sum()

    # firm sizes
    area = ax.fill_between(
        x=mcaps.index + 1,
        y1=mcaps,
        # y2=1e0,
        label="Asset market capitalization",
        alpha=0.7,
        # linewidth=1,
        # edgecolor="k",
        # hatch="|",
    )
    ax.scatter(
        mcaps.index + 1,
        mcaps,
        marker=".",
        color=colors[0],
        s=5,
    )
    scat = ax.scatter(
        mcaps.index[:100] + 1,
        mcaps[:100],
        label="100 largest assets",
        marker="x",
        color=colors[1],
    )

    ax.set_ylabel("Market capitalization")  # ('000 USD)")
    ax.set_xlabel("Size Rank")
    # ax.set_ylim([0, mcaps.max()*1.05])
    ax.set_xlim([-10, len(mcaps) + 10])
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="-")

    # cumulative
    line = ax2.plot(cumulative, label="Cumulative share (right axis)", color=colors[2])
    ax2.set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    ax2.set_yticklabels([f"{int(tick*100)}%" for tick in ax2.get_yticks()])
    ax2.grid(True, color="gray")
    ax2.set_ylabel("Cumulative share")
    ax2.set_ylim([0, 1.01])

    # cutoffs
    for pct in ax2.get_yticks()[1:-1]:
        x = cumulative[cumulative.gt(pct)].index[0]
        ax2.scatter(x=x, y=cumulative[x], marker="o", color=colors[2])
        ax2.text(
            x=x + 10,
            y=cumulative[x] - 0.04,
            s=f"{x} assets: {cumulative[x]*100:.2f}% of total market capitalization",
            color=colors[2],
        )

    # legend
    elements = [area, scat, line[0]]
    labels = [e.get_label() for e in elements]
    ax.legend(elements, labels)  # , bbox_to_anchor=(1.05, 0.5), loc="center left")

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
