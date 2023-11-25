# # Roliing Granular Estimates
# ## Imports

# %load_ext autoreload
# %autoreload 2

import numpy as np
from euraculus.data.map import DataMap
from euraculus.utils.plot import save_ax_as_pdf
from euraculus.settings import (
    OUTPUT_DIR,
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    AMEX_INCLUSION_DATE,
    NASDAQ_INCLUSION_DATE,
    SPLIT_DATE,
    TIME_STEP,
    ESTIMATION_WINDOW,
    SECTOR_COLORS,
    FIRST_ESTIMATION_DATE,
)
from kungfu.plotting import add_recession_bars
import kungfu as kf

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

save_outputs = True

# ## Load & prepare data

# %%time
data = DataMap(DATA_DIR)
df_estimates = data.read("analysis/df_estimates.pkl")

df_estimates["ncusip"][FIRST_ESTIMATION_DATE:].unstack().notna().sum(axis=1)#.to_csv(f"{OUTPUT_DIR}/sampled_assets.csv")

df_estimates["ticker"][FIRST_ESTIMATION_DATE:].unstack().notna().sum(axis=1)#.to_csv(f"{OUTPUT_DIR}/sampled_assets.csv")

# ## Granular analysis

# +
df_in = df_estimates["fevd_in_connectedness"].unstack()[SPLIT_DATE:]
df_in = df_in.loc[:, df_in.notna().sum()>0]

fig, ax = plt.subplots(1, 1, figsize=(20, 5*1.2))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
ax.set_ylim([0, 0.5])
ax.set_ylabel("FEVD In-connectedness")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 0.25:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 0.005,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/fevd_in_connectedness.pdf")

# +
df_in = df_estimates["fevd_out_connectedness"].unstack()[SPLIT_DATE:]
df_in = df_in.loc[:, df_in.notna().sum()>0]

fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
ax.set_ylim([0, 1.3])
ax.set_ylabel("FEVD Out-connectedness")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 0.25:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 0.005,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/fevd_out_connectedness.pdf")

# +
df_in = df_estimates["wfevd_out_connectedness"].unstack()[SPLIT_DATE:]
df_in = df_in.loc[:, df_in.notna().sum()>0]

fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
# ax.set_ylim([0, 1.3])
ax.set_ylabel("wFEVD Out-connectedness")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 0.5:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 0.005,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/wfevd_out_connectedness.pdf")

# +
df_in = df_estimates["wfevd_out_connectedness"].unstack()[SPLIT_DATE:].div(df_estimates["wfevd_out_connectedness"].groupby("sampling_date").sum()[SPLIT_DATE:].values.reshape(-1, 1))/0.01
df_in = df_in.loc[:, df_in.notna().sum()>0]

fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
# ax.set_ylim([0, 1.3])
ax.set_ylabel("wFEVD Out-connectedness weight factor")
df_events = pd.DataFrame(columns=["ticker", "company", "sector", "peak"], index=pd.Index([], name="date"))

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 2:
        max_pos = series.argmax()
        
        date=df_in.index[max_pos]
        df_events.at[date, "ticker"] = ticker
        df_events.at[date, "company"] = df_estimates.loc[(series.dropna().index[-1], permno), "comnam"]
        df_events.at[date, "sector"] = df_estimates.loc[(series.dropna().index[-1], permno), "ff_sector_ticker"]
        df_events.at[date, "peak"] = max_value
        
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 0.005,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
ax.axhline(1, color="k", linestyle="-", linewidth=1)
plt.yticks(list(plt.yticks()[0]) + [1])
ax.set_ylim([0, 10])
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/wfevd_out_connectedness_weight.pdf")
# -

df_events["company"] = df_events["company"].str.title().str.replace("&", "\&")
df_events["peak"] = df_events["peak"].astype(float).round(2)
df_events = df_events.sort_index()
df_events.index = df_events.index.strftime('%Y-%m')
df_events.to_latex(
    buf=OUTPUT_DIR / "analysis" / "network_events_table.tex",
    multirow=False,
    multicolumn_format ='c',
    na_rep='',
    escape=False,
)
df_events

sector_table = pd.Series({
    'NoDur': 'Consumer Nondurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys',
 'Durbl': 'Consumer Durables -- Cars, TVs, Furniture, Household Appliances',
 'Manuf': 'Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing',
 'Enrgy': 'Oil, Gas, and Coal Extraction and Products',
 'Chems': 'Chemicals and Allied Products',
 'BusEq': 'Business Equipment -- Computers, Software, and Electronic Equipment',
    'Telcm': 'Telephone and Television Transmission',
 'Utils': 'Utilities',
 'Shops': 'Wholesale, Retail, and Some Services (Laundries, Repair Shops)',
 'Hlth': 'Healthcare, Medical Equipment, and Drugs',
 'Money': 'Finance',
 'Other': 'Other -- Mines, Constr, BldMt, Trans, Hotels, Bus Serv, Entertainment',
}, name="Sector").to_frame()
sector_table.index.name = "Sector ticker"
sector_table.to_latex(
    buf=OUTPUT_DIR / "analysis" / "sector_table.tex",
    multirow=False,
    multicolumn_format ='c',
    na_rep='',
    escape=False,
)
sector_table

# +
network_weights = (
    df_estimates.groupby(["sampling_date", "ff_sector_ticker"])[
        "wfevd_out_connectedness"
    ]
    .sum()
    .groupby("sampling_date")
    .apply(lambda x: x/x.sum())
    .unstack()
)[reversed(list(SECTOR_COLORS.keys()))]

fig, ax = plt.subplots(figsize=(20, 8))

ax.stackplot(
    network_weights.index,
    (network_weights).fillna(0).iloc[:, :].values.T,
    labels=network_weights.columns,
    alpha=1.0,
    colors=reversed(plt.get_cmap("Paired")(np.linspace(0, 1, 12))),
)
ax.set_xlim([SPLIT_DATE, network_weights.index[-1]])
ax.set_ylim([0, 1])
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    reversed(handles),
    reversed(labels),
    title="Sectors",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)

# ax.set_title("Network influence decomposition by sector")
ax.set_xlabel("Date")
ax.set_ylabel("Sector weight share")
ax.set_axisbelow(False)
ax.grid(True, linestyle=":", axis="y")
ax.grid(False, axis="x")

add_recession_bars(
    ax, freq="M", startdate=network_weights.index[0], enddate=network_weights.index[-1]
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/wfevd_out_connectedness_sector_weight.pdf")

# +
network_weights = (
    df_estimates.groupby(["sampling_date", "ff_sector_ticker"])[
        "mean_mcap"
    ]
    .sum()
    .groupby("sampling_date")
    .apply(lambda x: x/x.sum())
    .unstack()
)[reversed(list(SECTOR_COLORS.keys()))]

fig, ax = plt.subplots(figsize=(24, 10))

ax.stackplot(
    network_weights.index,
    (network_weights).fillna(0).iloc[:, :].values.T,
    labels=network_weights.columns,
    alpha=1.0,
    colors=reversed(plt.get_cmap("Paired")(np.linspace(0, 1, 12))),
)
ax.set_xlim([SPLIT_DATE, network_weights.index[-1]])
ax.set_ylim([0, 1])
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    reversed(handles),
    reversed(labels),
    title="Sectors",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)

ax.set_title(
    f"Network influence decomposition by sector"
)
ax.set_xlabel("Date")
ax.set_ylabel("Weight share")
ax.set_axisbelow(False)
ax.grid(True, linestyle=":", axis="y")
ax.grid(False, axis="x")

add_recession_bars(
    ax, freq="M", startdate=network_weights.index[0], enddate=network_weights.index[-1]
)

# +
df_in = df_estimates["wfevd_full_out_connectedness"].unstack()[SPLIT_DATE:]
df_in = df_in.loc[:, df_in.notna().sum()>0]

fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
# ax.set_ylim([0, 1.3])
ax.set_ylabel("Market impact")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 2:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 0.005,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/wfevd_full_out_connectedness.pdf")

# +
df_in = df_estimates["mean_mcap"].unstack()[SPLIT_DATE:].div(df_estimates["mean_mcap"].groupby("sampling_date").sum()[SPLIT_DATE:].values.reshape(-1, 1))/0.01
df_in = df_in.loc[:, df_in.notna().sum()>0]

fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
# ax.set_ylim([0, 1.3])
ax.set_ylabel("Weight impact factor")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 2:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 0.005,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/mean_mcap.pdf")

# +
# df_in = df_estimates[SPLIT_DATE:]["wfevd_out_connectedness"].unstack()/(df_estimates[SPLIT_DATE:]["mean_mcap"].unstack().div(df_estimates[SPLIT_DATE:]["mean_mcap"].groupby("sampling_date").sum().values.reshape(-1, 1))*100)
df_in = (df_estimates[SPLIT_DATE:]["wfevd_out_connectedness"].unstack().div(df_estimates[SPLIT_DATE:]["wfevd_out_connectedness"].groupby("sampling_date").sum().values.reshape(-1, 1))*100)- (df_estimates[SPLIT_DATE:]["mean_mcap"].unstack().div(df_estimates[SPLIT_DATE:]["mean_mcap"].groupby("sampling_date").sum().values.reshape(-1, 1))*100)
fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
# ax.set_ylim([0, 1.3])
ax.set_ylabel("Amplifier")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 1.5:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 0.005,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/amplifier.pdf")

# +
df_in = df_estimates[SPLIT_DATE:]["wfevd_full_out_connectedness"].unstack()/(df_estimates[SPLIT_DATE:]["mean_mcap"].unstack().div(df_estimates[SPLIT_DATE:]["mean_mcap"].groupby("sampling_date").sum().values.reshape(-1, 1))*100)
fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
# ax.set_ylim([0, 1.3])
ax.set_ylabel("Amplifier")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 1.5:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 0.005,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/amplifier.pdf")
# -



# +
df_in = df_estimates[SPLIT_DATE:]["grp"].unstack() *250
fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
# ax.set_ylim([0, 1.3])
ax.set_ylabel("GRP")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 0.001:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 1e-5,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/grp.pdf")

# +
df_in = df_estimates[SPLIT_DATE:]["grpf"].unstack() *250
fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_in.index[0], df_in.index[-1]])
# ax.set_ylim([0, 1.3])
ax.set_ylabel("GRP (full D)")

for permno, series in df_in.iteritems():
    # get tickers
    ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_in.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_estimates.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]

    # label extremes
    max_value = series.max()
    if max_value >= 1e-3:
        max_pos = series.argmax()
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=1.5,
            alpha=1,
        )
        ax.text(
            series.index[max_pos],
            max_value + 1e-5,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

    # other assets
    else:
        ax.plot(
            series,
            color=SECTOR_COLORS[ff_sector_ticker],
            linewidth=0.5,
            alpha=0.5,
        )

# other elements
add_recession_bars(
    ax, startdate=df_in.index[0], enddate=df_in.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/grpf.pdf")
# -



# +
df_prem = df_estimates[SPLIT_DATE:]["grp"].unstack()
fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim([df_prem.index[0], df_prem.index[-1]])
# ax.set_ylim([0, 0.05])
ax.set_ylabel("Granular Risk Premia")

for permno, series in df_prem.iteritems():
    # get tickers
    ticker = df_premia.loc[(series.dropna().index[-1], permno), "ticker"]
    if type(ticker) != str:
        comnam = df_prem.loc[(series.dropna().index[-1], permno), "comnam"]
        ticker = "".join([s[0] for s in comnam.split(" ")])
    ff_sector_ticker = df_premia.loc[
        (series.dropna().index[-1], permno), "ff_sector_ticker"
    ]
    
    ax.plot(
        series,
        color=SECTOR_COLORS[ff_sector_ticker],
        linewidth=1.5,
        alpha=1,
    )

    # label extremes
    max_value = series.max()
    if max_value >= 0.005:
        max_pos = series.argmax()
        ax.text(
            series.index[max_pos],
            max_value + 5e-4,
            s=ticker,
            horizontalalignment="center",
            fontsize=8,
        )
        ax.scatter(
            series.index[max_pos],
            max_value,
            color=SECTOR_COLORS[ff_sector_ticker],
        )

# other elements
add_recession_bars(
    ax, startdate=df_prem.index[0], enddate=df_prem.index[-1]
)
sector_legend = ax.legend(
    handles=[
        mpl.patches.Patch(facecolor=color, label=ticker)
        for ticker, color in SECTOR_COLORS.items()
    ],
    title="Sectors",
    edgecolor="k",
    facecolor="lightgray",
    bbox_to_anchor=(1, 0.5),
    loc="center left",
)
# # if save_outputs:
# #     save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/fevd_out_connectedness.pdf")
# -



def plot_net_connectedness(
    df_estimates: pd.DataFrame,
    upper_threshold: float = 0.0,
    lower_threshold: float = 0.0,
):
    """"""
    # create plot
    df_net = df_estimates["fevd_in_connectedness"].unstack()
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_xlim([df_net.index[0], df_net.index[-1]])
    ax.set_ylim([0, 1.3])
    # ax.set_title(
    #     "Net-connectedness of tickers in the rolling FEVD spillover networks"
    # )
    ax.set_ylabel("FEVD Out-connectedness")
   
    for permno, series in df_net.iteritems():
        # get tickers
        ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
        if type(ticker) != str:
            comnam = df_estimates.loc[(series.dropna().index[-1], permno), "comnam"]
            ticker = "".join([s[0] for s in comnam.split(" ")])
        ff_sector_ticker = df_estimates.loc[
            (series.dropna().index[-1], permno), "ff_sector_ticker"
        ]
        
        # find extremes
        max_value = series.max()
        min_value = series.min()
        
        # influencers
        if max_value >= upper_threshold:
            max_pos = series.argmax()
            ax.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=1.5,
                alpha=1,
            )
            ax.text(
                series.index[max_pos],
                max_value + 0.015,
                s=ticker,
                horizontalalignment="center",
                fontsize=8,
            )
            ax.scatter(
                series.index[max_pos],
                max_value,
                color=SECTOR_COLORS[ff_sector_ticker],
            )
            
        # followers
        if min_value <= lower_threshold:
            min_pos = series.argmin()
            ax.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=1.5,
                alpha=1,
            )
            ax.text(
                series.index[min_pos],
                min_value - 0.015,
                s=ticker,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=7.5,
            )
            ax.scatter(
                series.index[min_pos],
                min_value,
                color=SECTOR_COLORS[ff_sector_ticker],
            )
            
        # other assets
        if max_value < upper_threshold and min_value > lower_threshold:
            ax.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=0.5,
                alpha=0.5,
            )

    # other elements
    add_recession_bars(
        ax, startdate=df_net.index[0], enddate=df_net.index[-1]
    )
    sector_legend = ax.legend(
        handles=[
            mpl.patches.Patch(facecolor=color, label=ticker)
            for ticker, color in SECTOR_COLORS.items()
        ],
        title="Sectors",
        edgecolor="k",
        facecolor="lightgray",
        bbox_to_anchor=(1, 0.5),
        loc="center left",
    )
    return ax


ax = plot_net_connectedness(df_estimates[SPLIT_DATE:], upper_threshold=0.2, lower_threshold=-0.1)
# if save_outputs:
#     save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/fevd_out_connectedness.pdf")



def plot_net_connectedness(
    df_estimates: pd.DataFrame,
    upper_threshold: float = 0.0,
    lower_threshold: float = 0.0,
):
    """"""
    # create plot
    df_net = df_estimates["wfevd_net_connectedness"].unstack()    
    # df_net = (df_estimates["wfevd_full_out_connectedness"]/df_estimates["wfevd_full_in_connectedness"]).unstack()
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_xlim([df_net.index[0], df_net.index[-1]])
    ax.set_title(
        "Net-connectedness of tickers in the rolling FEVD spillover networks"
    )
    ax.set_ylabel("Net-connectedness")
   
    for permno, series in df_net.iteritems():
        # get tickers
        ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
        if type(ticker) != str:
            comnam = df_estimates.loc[(series.dropna().index[-1], permno), "comnam"]
            ticker = "".join([s[0] for s in comnam.split(" ")])
        ff_sector_ticker = df_estimates.loc[
            (series.dropna().index[-1], permno), "ff_sector_ticker"
        ]
        
        # find extremes
        max_value = series.max()
        min_value = series.min()
        
        # influencers
        if max_value >= upper_threshold:
            max_pos = series.argmax()
            ax.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=1.5,
                alpha=1,
            )
            ax.text(
                series.index[max_pos],
                max_value + 0.015,
                s=ticker,
                horizontalalignment="center",
                fontsize=8,
            )
            ax.scatter(
                series.index[max_pos],
                max_value,
                color=SECTOR_COLORS[ff_sector_ticker],
            )
            
        # followers
        if min_value <= lower_threshold:
            min_pos = series.argmin()
            ax.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=1.5,
                alpha=1,
            )
            ax.text(
                series.index[min_pos],
                min_value - 0.015,
                s=ticker,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=7.5,
            )
            ax.scatter(
                series.index[min_pos],
                min_value,
                color=SECTOR_COLORS[ff_sector_ticker],
            )
            
        # other assets
        if max_value < upper_threshold and min_value > lower_threshold:
            ax.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=0.5,
                alpha=0.5,
            )

    # other elements
    add_recession_bars(
        ax, startdate=df_net.index[0], enddate=df_net.index[-1]
    )
    sector_legend = ax.legend(
        handles=[
            mpl.patches.Patch(facecolor=color, label=ticker)
            for ticker, color in SECTOR_COLORS.items()
        ],
        title="Sectors",
        edgecolor="k",
        facecolor="lightgray",
        bbox_to_anchor=(1, 0.5),
        loc="center left",
    )
    ax.set_ylim([0, 1])
    # ax.set_yscale('symlog')
    return ax


def plot_net_connectedness(
    df_estimates: pd.DataFrame,
    upper_threshold: float = 0.0,
    lower_threshold: float = 0.0,
):
    """"""
    # create plot
    df_net = df_estimates["wfevd_net_connectedness"].unstack()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(20, 14), gridspec_kw={"height_ratios": [2, 1]})
    ax.set_xlim([df_net.index[0], df_net.index[-1]])
    ax2.set_xlim([df_net.index[0], df_net.index[-1]])
    # ax.set_title(
    #     "Net-connectedness of tickers in the rolling FEVD spillover networks"
    # )
    ax.set_ylabel("wFEVD Net-connectedness")
    ax.yaxis.set_label_coords(-0.03, 0)
   
    for permno, series in df_net.iteritems():
        # get tickers
        ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
        if type(ticker) != str:
            comnam = df_estimates.loc[(series.dropna().index[-1], permno), "comnam"]
            ticker = "".join([s[0] for s in comnam.split(" ")])
        ff_sector_ticker = df_estimates.loc[
            (series.dropna().index[-1], permno), "ff_sector_ticker"
        ]
        
        # find extremes
        max_value = series.max()
        min_value = series.min()
        
        # influencers
        if max_value >= upper_threshold:
            max_pos = series.argmax()
            ax.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=1.5,
                alpha=1,
            )
            ax.text(
                series.index[max_pos],
                max_value + 0.015,
                s=ticker,
                horizontalalignment="center",
                fontsize=8,
            )
            ax.scatter(
                series.index[max_pos],
                max_value,
                color=SECTOR_COLORS[ff_sector_ticker],
            )
            
        # followers
        if min_value <= lower_threshold:
            min_pos = series.argmin()
            ax2.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=1.5,
                alpha=1,
            )
            ax2.text(
                series.index[min_pos],
                min_value - 0.015 * 8,
                s=ticker,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=7.5,
            )
            ax2.scatter(
                series.index[min_pos],
                min_value,
                color=SECTOR_COLORS[ff_sector_ticker],
            )
            
        # other assets
        if max_value < upper_threshold and min_value > lower_threshold:
            ax.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=0.5,
                alpha=0.5,
            )
            ax2.plot(
                series,
                color=SECTOR_COLORS[ff_sector_ticker],
                linewidth=0.5,
                alpha=0.5,
            )

    # other elements
    add_recession_bars(
        ax, startdate=df_net.index[0], enddate=df_net.index[-1]
    )
    add_recession_bars(
        ax2, startdate=df_net.index[0], enddate=df_net.index[-1]
    )
    sector_legend = ax.legend(
        handles=[
            mpl.patches.Patch(facecolor=color, label=ticker)
            for ticker, color in SECTOR_COLORS.items()
        ],
        title="Sectors",
        edgecolor="k",
        facecolor="lightgray",
        bbox_to_anchor=(1, 0),
        loc="center left",
    )
    ax.set_ylim(0, 1)
    ax2.set_ylim(-4, 0)
    
    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    d = 0.005  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    fig.subplots_adjust(hspace=0.0)
    
    ax.axhline(0, color="k", linewidth=1.5, linestyle="--")
    ax2.axhline(0, color="k", linewidth=1.5, linestyle="--")
    
    # ax.set_yscale('symlog')
    return ax


ax = plot_net_connectedness(df_estimates[SPLIT_DATE:], upper_threshold=0.2, lower_threshold=-0.2)
# if save_outputs:
#     save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/wfevd_net_connectedness.pdf")

# ticker lookup
data.lookup_ticker(
    tickers=["WYE", "FOXA", "WMX", "XTO", "WB", "ENE", "GME", "BA"],
    date=None,
)


