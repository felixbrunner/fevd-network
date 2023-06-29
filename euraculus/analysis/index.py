import pandas as pd
import numpy as np


def construct_long_weights(
    df_estimates: pd.DataFrame,
    variable: str = "equal",
) -> pd.Series:
    """"""
    if variable == "equal":
        weights = (
            pd.Series(1, index=df_estimates.index)
            .groupby("sampling_date")
            .transform(lambda x: 1 / x.count())
            .rename("weight")
            .replace(np.inf, np.nan)
        )

    else:
        weights = (
            df_estimates[variable]
            .groupby("sampling_date")
            .apply(lambda x: x / x.sum())
            .rename("weight")
            .replace(np.inf, np.nan)
        )

    return weights


def construct_residual_weights(
    df_estimates: pd.DataFrame,
    variable: str = "mean_mcap",
    benchmark_variable: str = "equal",
    constant_leverage: bool = False,
) -> pd.Series:
    """"""
    variable_weights = construct_long_weights(
        df_estimates=df_estimates,
        variable=variable,
    )
    benchmark_weights = construct_long_weights(
        df_estimates=df_estimates,
        variable=benchmark_variable,
    )
    residual_weights = variable_weights - benchmark_weights

    if constant_leverage:
        residual_weights = (
            residual_weights.groupby("sampling_date")
            .apply(lambda x: 2 * x / (x.abs().sum()))
            .replace(np.inf, np.nan)
        )

    return residual_weights


def calculate_excess_returns(
    df_returns: pd.DataFrame,
    df_rf: pd.DataFrame,
    return_variable: str = "retadj",
) -> pd.Series:
    """"""
    df_returns = df_returns.merge(df_rf, left_on="date", right_index=True, how="left")
    excess_returns = pd.Series(
        data=df_returns[return_variable] - df_returns["rf"], name="excess_return"
    )
    excess_returns = excess_returns.to_frame().join(df_returns["sampling_date"])
    return excess_returns


def build_index_portfolio(
    weights: pd.Series,
    returns: pd.Series,
    leg: str = "both",
) -> pd.Series:
    """"""
    df_combined = (
        returns.fillna(0).reset_index()
        .merge(weights.reset_index(), on=["sampling_date", "permno"], how="inner")
        .set_index(["date", "permno"])
        .drop(columns=["sampling_date"])
    )

    if leg == "both":
        index = df_combined.prod(axis=1).groupby("date").sum().rename("long_short")
    elif leg == "long":
        index = (
            df_combined[df_combined["weight"] > 0]
            .prod(axis=1)
            .groupby("date")
            .sum()
            .rename("long")
        )
    elif leg == "short":
        index = (
            df_combined[df_combined["weight"] < 0]
            .prod(axis=1)
            .groupby("date")
            .sum()
            .rename("short")
        )
    else:
        raise ValueError("unknown leg type")

    return index
