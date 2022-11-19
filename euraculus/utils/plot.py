import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as mno
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.decomposition import PCA
from euraculus.settings import COLORS

import kungfu as kf
from kungfu.plotting import add_recession_bars

# %% Plot settings

# style
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = [17, 8]

# colors
# DARK_BLUE = "#014c63"
# DARK_GREEN = "#3b5828"
# DARK_RED = "#880000"
# ORANGE = "#ae3e00"
# DARK_YELLOW = "#ba9600"
# PURPLE = "#663a82"
# COLORS = [DARK_BLUE, DARK_GREEN, DARK_RED, DARK_YELLOW, ORANGE, PURPLE]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS)


def distribution_plot(
    data: pd.Series,
    bins: int = 100,
    title: str = "Data distribution",
    show_kde: bool = True,
    show_gaussian: bool = False,
    drop_tails: float = 0,
    ax=None,
):
    """Create a histogram of the input data.

    Args:
        data: Data as a pandas Series.
        bins: Number of bins to plot the histogram.
        title: Title for the figure.
        show_kde: Indicates to show Gaussian kernel density estimate.
        show_gaussian: Indicates to show Gaussian distribution.
        drop_tails: Fraction of outliers to drop as a decimal,
            e.g., 0.01 to drop the 1% most unlikely observations.
        ax: Matplotlib Axis object to plot into (optional).

    Returns:
        ax: Matplotlib Axis object.
    """
    # prepare plot
    if ax is None:
        ax = plt.gca()

    # drop outliers
    data = data[
        (data.quantile(0 + drop_tails / 2) < data)
        & (data < data.quantile(1 - drop_tails / 2))
    ]

    # plot histogram, kernel density estimate and mean
    ax.hist(data, bins=bins, label="Data")
    xvals = np.linspace(*ax.get_xlim(), bins)
    ax.axvline([data.mean()], color="grey", label="Sample mean", linestyle="--")
    if show_kde:
        kde = sp.stats.gaussian_kde(data.squeeze())
        ax.plot(
            xvals,
            kde(xvals)
            * (data.max() - data.min())
            / bins
            * len(data)
            * (1 - drop_tails),
            label="Scaled KDE",
            c="k",
        )
    if show_gaussian:
        normal = sp.stats.norm(data.mean(), data.std())
        ax.plot(
            xvals,
            normal.pdf(xvals)
            * (data.max() - data.min())
            / bins
            * len(data)
            * (1 - drop_tails),
            label="Gaussian PDF",
            c="orange",
        )

    # formatting
    ax.set_title(title)
    ax.legend()

    return ax


def save_ax_as_pdf(
    ax,
    save_path: str,
):
    """Save a figure given an axis object as a PDF file.

    Args:
        ax: Matplotlib Axis object to save.
        save_path: Path to save figure into.
    """
    fig = ax.get_figure()
    fig.savefig(
        save_path,
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"Plot saved at '{save_path}'")


def missing_data_matrix(
    data: pd.DataFrame,
    title: str = "Missing Data",
):
    """Creates and saves a missingno plot.

    Args:
        data: Dataframe to show missing data of.
        title: Title of the plot.

    Returns:
        ax: Matplotlib Axis object.
    """
    # plot
    ax = mno.matrix(data, labels=False)

    # label
    ax.set_title(title, fontsize=16)
    plt.xticks(np.arange(data.shape[1]), data.columns, rotation=90, fontsize=12)

    return ax


def matrix_heatmap(
    data: np.ndarray,
    title: str = "Correlation Matrix",
    labels: list = None,
    secondary_labels: list = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "seismic",
    infer_limits: bool = False,
    infer_vmax: bool = False,
    ax=None,
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
        infer_limits: Indicates if heatmap range is inferred, default=False.
        infer_vmax: Indicates if heatmap maximum is inferred, default=False.
        ax: Matplotlib Axis object to plot into (optional).

    Returns:
        ax: Matplotlib Axis object.

    """
    # prepare plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

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
    # fig, ax = plt.subplots(1, 1, figsize=(12, 10))
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

    return ax


def draw_network(
    network,
    df_info,
    title: str = "Network",
    save_path: str = None,
    pos: dict = None,
    **kwargs,
) -> dict:
    """Draw FEVD as a force layout network plot.

    Args:
        network: Network object to plot.
        df_info: Accompanying information in a dataframe.
        title: Title of the plot, default="Network".
        save_path: Path to save the figure.
        pos: Dictionary with initial locations for all/some nodes.

    Returns:
        layout: A dictionary of positions keyed by node.
    """
    # set up graph
    ticker_dict = {i: tick for (i, tick) in enumerate(df_info["ticker"].values)}
    g = network.to_graph()
    g = nx.relabel_nodes(g, ticker_dict)
    table = network.adjacency_matrix

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
    net_connectedness = network.net_connectedness()
    include_edges = table > np.percentile(table, q=90, axis=None, keepdims=True)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=[22, 12])
    ax.set_title(title)
    ax.grid(False)

    # draw nodes
    node_options = {
        "node_size": 750
        + (
            df_info["mean_mcap_volatility"] / df_info["mean_mcap_volatility"].mean()
        ).values
        * 250,
        "node_color": [
            sector_colors[ff_sector_codes[i]]
            for i in df_info["ff_sector_ticker"].values
        ],
        "linewidths": [
            4
            if nc > np.percentile(net_connectedness, 80)
            else 4
            if nc < np.percentile(net_connectedness, 20)
            else 0
            for nc in net_connectedness
        ],
        "alpha": 0.9,
        "edgecolors": [
            "w"
            if nc > np.percentile(net_connectedness, 80)
            else "grey"
            if nc < np.percentile(net_connectedness, 20)
            else "r"
            for nc in net_connectedness
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
                label="Influencers",
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
                label="Followers",
                markersize=10,
                linewidth=0,
            ),
        ],
        title="Most connected assets",
        loc="lower right",
        edgecolor="k",
    )
    ax.add_artist(sector_legend)
    ax.add_artist(influence_legend)

    # save
    if save_path:
        fig.savefig(save_path, format="png", dpi=fig.dpi, bbox_inches="tight")

    return layout


def contribution_bars(
    scores: np.ndarray,
    names: list,
    title: str = None,
    normalize: bool = False,
    ax=None,
):
    """Plot proportional contributions as a bar plot.

    Args:
        scores: Proportion defining data array.
        names: Names of the values to be shown in the plot.
        title: Totle for the plot.
        normalize: Indicates if values should sum to one.
        ax: Matplotlib Axis object to plot into (optional).

    Returns:
        ax: Matplotlib Axis object.
    """
    # prepare plot
    if ax is None:
        ax = plt.gca()

    # prepare data
    data = pd.Series(data=scores, index=names).sort_values(ascending=False)
    if normalize:
        data /= data.sum()

    # plot
    ax.bar(x=np.arange(1, len(data) + 1), height=data)
    ax.set_title(title + (" (normalized to one)" if normalize else ""))

    # format
    ax.set_xlim([0, 101])
    ax.set_xticks(np.arange(1, len(scores) + 1), minor=False)
    ax.set_xticklabels(data.index, rotation=90, minor=False)
    ax.tick_params(axis="x", which="major", bottom=True, labelbottom=True)

    return ax


###################################################################################

# def cov_cv_contour(cv, levels=12, logx=False, logy=False, save_path=None):
#     """Creates a countour plot from cross-validation
#     for hyperparamter search.
#     """
#     # create plot
#     fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#     ax.set_title("Adaptive Threshold Estimation Hyper-Parameter Search Grid")

#     # data
#     x_name, y_name = cv.param_grid.keys()
#     x_values, y_values = cv.param_grid.values()
#     x_grid, y_grid = np.meshgrid(x_values, y_values)
#     z_values = (
#         -cv.cv_results_["mean_test_score"].reshape(len(x_values), len(y_values)).T
#     )

#     # contour plotting
#     contour = ax.contourf(
#         x_grid,
#         y_grid,
#         z_values,
#         levels=levels,
#         cmap="RdYlGn_r",
#         antialiased=True,
#         alpha=1,
#     )
#     ax.contour(
#         x_grid,
#         y_grid,
#         z_values,
#         levels=levels,
#         colors="k",
#         antialiased=True,
#         linewidths=1,
#         alpha=0.6,
#     )
#     cb = fig.colorbar(contour)

#     # grid & best estimator
#     x_v = [a[x_name] for a in cv.cv_results_["params"]]
#     y_v = [a[y_name] for a in cv.cv_results_["params"]]
#     ax.scatter(x_v, y_v, marker=".", label="grid", color="k", alpha=0.25)
#     ax.scatter(
#         *cv.best_params_.values(), label="best estimator", marker="x", s=100, color="k"
#     )

#     # labels & legend
#     ax.set_xlabel("$\delta$ (0.5=sample cov, 1=zeros)")  # x_name)
#     ax.set_ylabel('$\eta$ (0=zeros, 1=soft-thresholding, 2="ridge")')  # y_name)
#     ax.legend()  # loc='upper left')
#     cb.set_label("Cross-Validation Loss", rotation=90)
#     if logx:
#         ax.set_xscale("log")
#     if logy:
#         ax.set_yscale("log")

#     # limits
#     ax.set_xlim([min(x_values), max(x_values)])
#     ax.set_ylim([min(y_values), max(y_values)])

#     # save
#     if save_path:
#         fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


# def cov_scatter_losses(cv, save_path=None):
#     # extract data
#     train_losses = -cv.cv_results_["mean_train_score"]
#     valid_losses = -cv.cv_results_["mean_test_score"]
#     deltas = pd.Series([d["delta"] for d in cv.cv_results_["params"]])
#     etas = pd.Series([d["eta"] for d in cv.cv_results_["params"]])
#     best = cv.best_index_

#     # figure parameters
#     fig, ax = plt.subplots(1, 1)
#     colors = deltas
#     sizes = (etas * 200) + 50

#     # labels
#     ax.set_xlabel("Mean Training Loss (In-sample)")
#     ax.set_ylabel("Mean Validation Loss (Out-of-sample)")
#     ax.set_title("Adaptive Threshold Estimation Cross-Validation Errors")

#     # scatter plots
#     sc = ax.scatter(
#         train_losses, valid_losses, c=colors, s=sizes, cmap="bone", edgecolor="k"
#     )
#     ax.scatter(
#         train_losses[best],
#         valid_losses[best],
#         s=sizes[best] * 2,
#         c="r",
#         edgecolor="k",
#         marker="x",
#         zorder=100,
#         label="best model",
#     )

#     # 45 degree line
#     x0, x1 = ax.get_xlim()
#     y0, y1 = ax.get_ylim()
#     lims = [max(x0, y0), min(x1, y1)]
#     ax.plot(lims, lims, color="grey", linestyle="--", label="45-degree line", zorder=0)

#     # legends
#     color_legend = ax.legend(
#         *sc.legend_elements(prop="colors", num=colors.nunique()),
#         loc="upper left",
#         title="δ",
#     )
#     ax.add_artist(color_legend)
#     handles, _ = sc.legend_elements(prop="sizes", alpha=0.6, num=sizes.nunique())
#     size_legend = ax.legend(
#         handles, [round(i, 2) for i in etas.unique()], loc="lower right", title="η"
#     )
#     ax.add_artist(size_legend)
#     ax.legend(loc="lower center")

#     # save
#     if save_path:
#         fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")


# def get_edge_colors(graph, percentage=5, col1="grey", col2="firebrick"):
#     imbalances = []
#     for i, j in graph.edges():
#         try:
#             ij_weight = graph[i][j]["weight"]
#             ji_weight = graph[j][i]["weight"]
#             imbalances += [abs(ij_weight - ji_weight)]
#         except:
#             pass
#     threshold = np.percentile(np.array(imbalances), 100 - percentage)
#     colors = [col2 if imb > threshold else col1 for imb in imbalances]
#     return colors


# def var_timeseries(idio_var, total_var=None, index_var=None, save_path=None):
#     """"""
#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     systematic = (total_var.mean(axis=1) - idio_var.mean(axis=1)).clip(0, None)

#     fig, ax = plt.subplots(1, 1)
#     ax.set_title("Variance decomposition: cross-sectional means")

#     ax.plot(total_var.mean(axis=1), label="Total variances", c=colors[0])

#     ax.plot(systematic, c=colors[1], linewidth=2, linestyle="-")
#     ax.fill_between(
#         idio_var.index,
#         0,
#         systematic,
#         alpha=0.5,
#         label="Systematic variance contribution",
#         color=colors[1],
#     )

#     ax.fill_between(
#         idio_var.index,
#         systematic,
#         total_var.mean(axis=1),
#         alpha=0.5,
#         label="Non-systematic variance contribution",
#         color=colors[0],
#     )

#     ax.plot(index_var, c=colors[2], label="SPY variance", linestyle="--", alpha=0.6)

#     ax.legend()
#     #     add_recession_bars(ax, startdate=idio_volas.index[0], enddate=idio_volas.index[-1])
#     if save_path:
#         fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
