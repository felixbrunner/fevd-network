"""
This module provides functionality to sample the largest companies from
CRSP data at the end of each month.
"""

import datetime as dt
from string import ascii_uppercase as ALPHABET

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from euraculus.data.map import DataMap


class LargeCapSampler:
    """Object to perform the sampling of the largest companies in the CRSP dataset."""

    def __init__(
        self,
        n_assets: int = 100,
        datamap: DataMap = None,
        back_offset: int = 12,
        forward_offset: int = 12,
    ):
        """Set up the sampler and link to file system.

        Args:
            n_assets: Number of largest companies to sample.
            data: DataMap to access local file system with raw data.
            back_offset: Length of period to consider before sampling date in months.
            forward_offset: Length of period to consider after sampling date in months.
        """
        self._data = datamap
        self.n_assets = n_assets
        self.back_offset = back_offset
        self.forward_offset = forward_offset

    @property
    def data(self) -> DataMap:
        """DataMap object to allow local data access."""
        if self._data:
            return self._data
        else:
            raise AttributeError("No datamap defined to load data")

    @data.setter
    def data(self, datamap):
        self._data = datamap

    @staticmethod
    def _prepare_date(dateinput: str) -> pd.Timestamp:
        """Transform input string into datetime object.

        Args:
            dateinput: Input date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.

        Returns:
            date: Transformed input as datetime object.
        """
        try:
            date = pd.to_datetime(dateinput)
        except:
            raise ValueError(
                "date '{}' could not be converted to datetime, try format 'YYYY-MM-DD'".format(
                    date
                )
            )

        return date

    def _load_sampling_data(self, sampling_date: pd.Timestamp) -> tuple:
        """Loads raw data from disk to perform sampling on.

        Args:
            sampling_date: Sampling date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.

        Returns:
            df_back: Sampled data of the period prior to the sampling date.
            df_forward: Sampled data of the period after the sampling date.
        """
        # format sampling date
        sampling_date = self._prepare_date(sampling_date)  # should come in right format

        # infer start date
        start_date = sampling_date - relativedelta(months=self.back_offset, days=-1)
        if start_date.month == 2 and start_date.day == 28:
            start_date += relativedelta(day=31)

        # infer end date
        end_date = sampling_date + relativedelta(months=self.forward_offset)
        if end_date.month == 2 and end_date.day == 28:
            end_date += relativedelta(day=31)

        # load data
        df_back = self.data.load_crsp_data(start_date, sampling_date)
        df_forward = self.data.load_crsp_data(
            sampling_date + dt.timedelta(days=1), end_date
        )
        return (df_back, df_forward)

    @staticmethod
    def _aggregate_permco(
        df: pd.DataFrame, selected_permnos: list = None
    ) -> pd.DataFrame:
        """Aggregate CRSP data to permco level.

        The selected data will be the security (permno) with the highest median variance per
        company (permco). A column 'permco_mcap' will be added with the aggregated market capitalization.

        Args:
            df: Dataframe with CRSP data.
            selected_permnos: Optional list of permnos to select.

        Returns:
            df_aggregated: Transformed dataframe.
        """

        # select permnos with largest median variance
        if selected_permnos is None:
            median_variances = df.groupby(["permco", "permno"])["var"].median()
            selected_permnos = median_variances.index.get_level_values("permno")[
                median_variances.groupby("permco").transform(max) == median_variances
            ].tolist()

        # aggregate
        df_aggregated = (
            df[df.index.get_level_values("permno").isin(selected_permnos)]
            .rename(columns={"mcap": "permno_mcap"})
            .reset_index()
        )
        aggregated_mcap = df.groupby(["date", "permco"])["mcap"].sum()
        df_aggregated = df_aggregated.merge(
            aggregated_mcap,
            how="left",
            on=["date", "permco"],
        ).set_index(["date", "permno"])

        return df_aggregated, selected_permnos

    @staticmethod
    def _has_all_days(df: pd.DataFrame, required_pct: float = 0.99) -> pd.Series:
        """Check data completeness.

        Return a Series of booleans to indicate whether permnos
        have more than 99% of days in dataframe.

        Args:
            df: CRSP data with column 'retadj'.
            required_pct: Percentage of required observations for completeness.

        Returns:
            has_all_days: Permno, bool pairs indicating completeness.
        """
        t_periods = df.index.get_level_values("date").nunique()
        has_all_days = df["retadj"].groupby("permno").count() > t_periods * required_pct
        has_all_days = has_all_days.rename("has_all_days")
        return has_all_days

    @staticmethod
    def _has_obs(df_back: pd.DataFrame, df_forward: pd.DataFrame) -> pd.Series:
        """Check if observations exist after sampling date.

        Return a Series of booleans to indicate whether there are subsequent
        observations.

        Args:
            df_back: CRSP data of the period prior to the sampling date.
            df_forward: CRSP data of the period after the sampling date.

        Returns:
            has_obs: Permno, bool pairs indicating subsequent observations.
        """
        permnos_back = df_back.index.get_level_values("permno").unique()
        permnos_forward = df_forward.index.get_level_values("permno").unique()
        has_obs = [permno in permnos_forward for permno in permnos_back]
        has_obs = pd.Series(has_obs, index=permnos_back, name="has_obs")
        return has_obs

    @staticmethod
    def _get_last_mcaps(df: pd.DataFrame) -> pd.Series:
        """Get last firm sizes from dataset.

        Create a series of last observed market capitalisation for
        all contained permnos.

        Args:
            df: CRSP data with column 'mcap'.

        Returns:
            last_mcap: Permno, mcap pairs with last observed sizes.
        """
        last_mcap = (
            df["mcap"]
            .unstack()
            .sort_index()
            .fillna(method="ffill", limit=1)
            .tail(1)
            .squeeze()
            .rename("last_mcap")
        )
        return last_mcap

    @staticmethod
    def _get_mean_mcaps(df: pd.DataFrame) -> pd.Series:
        """Get average firm sizes from dataset.

        Return a series of average market capitalisations for contained permnos.

        Args:
            df : CRSP data with column 'mcap'.

        Returns:
            mean_mcap: Permno, mcap pairs with average observed sizes.
        """
        mean_mcap = df["mcap"].unstack().mean().squeeze().rename("mean_mcap")
        return mean_mcap

    @staticmethod
    def _get_mean_mcap_volatility(df: pd.DataFrame) -> pd.Series:
        """Get average firm mcap volatilities from dataset.

        Return a series of average market capitalisations for contained permnos.

        Args:
            df : CRSP data with column 'mcap'.

        Returns:
            mean_mcap_volatility: Permno, mcap_volatility pairs with average observed vv.
        """
        mcap_volatility = df["mcap"] * np.sqrt(df["var"].fillna(df["noisevar"]))
        mean_mcap_volatility = (
            mcap_volatility.unstack().mean().squeeze().rename("mean_mcap_volatility")
        )
        return mean_mcap_volatility

    @staticmethod
    def _get_last_mcap_volatility(df: pd.DataFrame) -> pd.Series:
        """Get average firm mcap volatilities from dataset.

        Return a series of average market capitalisations for contained permnos.

        Args:
            df : CRSP data with column 'mcap'.

        Returns:
            mean_mcap_volatility: Permno, mcap volatilities pairs with average observed vv.
        """
        mcap_volatility = df["mcap"] * np.sqrt(df["var"].fillna(df["noisevar"]))
        last_mcap_volatility = (
            mcap_volatility.unstack()
            .sort_index()
            .fillna(method="ffill", limit=1)
            .tail(1)
            .squeeze()
            .rename("last_mcap_volatility")
        )
        return last_mcap_volatility
    
    @staticmethod
    def _get_value_vola(df: pd.DataFrame) -> pd.Series:
        """Get value times return volatility from dataset.

        Return a series of daily volatilities times value for contained permnos.

        Args:
            df : CRSP data with column 'retadj' and 'mcap'.

        Returns:
            valvola: Permno, valvola pairs with average observed sizes.
        """
        val = df["mcap"].unstack().mean().squeeze()
        vola = df["retadj"].unstack().std().squeeze()
        valvola = (val * vola).rename("value_vola")
        return valvola

    @staticmethod
    def _get_tickers(df: pd.DataFrame) -> pd.Series:
        """Get tickers for the data contained in dataframe.

        Return a series of tickers for all permos in the dataframe.

        Args:
            df: CRSP data with column 'ticker'.

        Returns:
            tickers: Permno, ticker pairs from dataframe.
        """
        tickers = df["ticker"].unstack().tail(1).squeeze().rename("ticker")
        return tickers

    def _get_company_names(self, df: pd.DataFrame) -> pd.Series:
        """Get company names for the data contained in dataframe.

        Return a series of company names for all permos in the dataframe.

        Args:
            df: CRSP data with permnos as indices

        Returns:
            comnams: Permno, company name pairs from dataframe.
        """
        # set up
        permnos = df.index.get_level_values("permno").unique().tolist()
        date = df.index.get_level_values("date").max()

        # lookup
        comnams = self._data.lookup_permnos(permnos=permnos, date=date).comnam
        company_names = pd.Series(index=permnos, data=comnams)
        return company_names

    def _describe_sampling_data(
        self,
        df_back: pd.DataFrame,
        df_forward: pd.DataFrame,
        characteristic: str = "mcap",
    ) -> pd.DataFrame:
        """Create summary statistics for the financial data provided.

        Return a summaring dataframe where the index consists
        of the permnos present in the dataframe.

        Args:
            df_back: CRSP data of the period prior to the sampling date.
            df_forward: CRSP data of the period after the sampling date.

        Returns:
            df_summary: Summarizing information in tabular form.
        """
        # assemble statistics
        df_summary = (
            self._get_tickers(df_back)
            .to_frame()
            .join(self._get_company_names(df_back))
            .join(self._has_all_days(df_back))
            .join(self._has_obs(df_back, df_forward).rename("has_next_obs"))
            .join(self._get_last_mcaps(df_back))
            .join(self._get_mean_mcaps(df_back))
            .join(self._get_last_mcap_volatility(df_back))
            .join(self._get_mean_mcap_volatility(df_back))
            .join(self._get_value_vola(df_back))
        )

        # set next obs to TRUE if there are no subsequent observations at all
        if not any(df_summary["has_next_obs"]):
            df_summary["has_next_obs"] = True

        # cross-sectional ranks
        df_summary["last_mcap_rank"] = (
            df_summary["last_mcap"]
            .where(df_summary["has_all_days"])
            .where(df_summary["has_next_obs"])
            .rank(ascending=False)
        )
        df_summary["mean_mcap_rank"] = (
            df_summary["mean_mcap"]
            .where(df_summary["has_all_days"])
            .where(df_summary["has_next_obs"])
            .rank(ascending=False)
        )
        df_summary["last_mcap_volatility_rank"] = (
            df_summary["mean_mcap_volatility"]
            .where(df_summary["has_all_days"])
            .where(df_summary["has_next_obs"])
            .rank(ascending=False)
        )
        df_summary["mean_mcap_volatility_rank"] = (
            df_summary["mean_mcap_volatility"]
            .where(df_summary["has_all_days"])
            .where(df_summary["has_next_obs"])
            .rank(ascending=False)
        )
        df_summary["mean_valvola_rank"] = (
            df_summary["value_vola"]
            .where(df_summary["has_all_days"])
            .where(df_summary["has_next_obs"])
            .rank(ascending=False)
        )

        return df_summary

    def _select_largest(
        self,
        df_summary: pd.DataFrame,
        method: str = "mean",
        characteristic: str = "mcap",
    ) -> list:
        """Pick N lagest companies.

        Return a list of permnos of the n companies
        with the highest market capitalisation end of year.

        Args:
            df_summary: Summary data to determine selection.
            method: Selection criterion, can be 'last' or 'mean' to select largest
                companies based on last or mean size.
            characteristic: Variable to perform ranking and selection on.

        Returns:
            permnos: Permnos of the N largest companies.
        """
        if not method in ["last", "mean"]:
            raise ValueError(
                "sorting method '{}' not supported, must be one of ['last', 'mean']".format(
                    method
                )
            )
        permnos = (
            df_summary.sort_values(f"{method}_{characteristic}_rank")
            .head(self.n_assets)
            .index.get_level_values("permno")
            .tolist()
        )
        return permnos

    @staticmethod
    def _make_tickers_unique(df: pd.DataFrame) -> pd.DataFrame:
        """Adds alphabetic suffixes to duplicate tickers.

        If tickers are not unique, add .<ALPHA> to duplicate tickers.

        Args:
            df: Multiindexed dataframe with column 'tickers'.

        Returns:
            df_: Transformed dataframe with unique 'tickers'.
        """
        df_ = df.copy()
        if isinstance(df_.index, pd.MultiIndex):
            tickers = df_["ticker"].unstack().iloc[-1, :]
        else:
            tickers = df_["ticker"]

        # find keys for each unique ticker
        for ticker in tickers.unique():
            ticker_keys = [
                key for (key, value) in tickers.iteritems() if value == ticker
            ]

            # append letter if duplicate
            if len(ticker_keys) > 1 and ticker is not None:
                for occurence, ticker_key in enumerate(ticker_keys):
                    rows = df_.index.get_level_values("permno") == ticker_key
                    df_.loc[rows, "ticker"] = ticker + "." + ALPHABET[occurence]

        return df_

    def sample(
        self, sampling_date: str, method: str = "mean", sampling_variable: str = "mcap"
    ) -> tuple:
        """Preprocess CRSP data to provide sample of large caps.

        Prepare asset return and variance datasets for a specified sampling date.

        Args:
           sampling_date: Sampling date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.
           sampling_variable: The variable to use for sample selection.

        Returns:
            df_historic: Sampled data of the period prior to the sampling date.
            df_future: Sampled data of the period after the sampling date.
            df_summary: Summary data used to determine selection.
        """
        # set up data
        sampling_date = self._prepare_date(sampling_date)
        df_historic, df_future = self._load_sampling_data(sampling_date)
        df_historic, selected_permnos = self._aggregate_permco(df_historic)
        if len(df_future) > 0:
            df_future, _ = self._aggregate_permco(
                df_future, selected_permnos=selected_permnos
            )

        # select assets
        df_summary = self._describe_sampling_data(df_historic, df_future)
        permnos = self._select_largest(
            df_summary, method=method, characteristic=sampling_variable
        )

        # slice
        df_historic = (
            df_historic[
                df_historic.index.isin(
                    pd.MultiIndex.from_product(
                        [
                            df_historic.index.get_level_values("date")
                            .unique()
                            .tolist(),
                            permnos,
                        ],
                        names=["date", "permno"],
                    )
                )
            ]
            .reindex(level="permno", labels=permnos)
            .dropna(how="all", subset=["mcap", "var", "noisevar", "retadj"])
        )
        df_historic = self._make_tickers_unique(df_historic)

        if not df_future.empty:
            df_future = (
                df_future[
                    df_future.index.isin(
                        pd.MultiIndex.from_product(
                            [
                                df_future.index.get_level_values("date")
                                .unique()
                                .tolist(),
                                permnos,
                            ],
                            names=["date", "permno"],
                        )
                    )
                ]
                .reindex(level="permno", labels=permnos)
                .dropna(how="all", subset=["mcap", "var", "noisevar", "retadj"])
            )
            df_future = self._make_tickers_unique(df_future)

        return (df_historic, df_future, df_summary)
