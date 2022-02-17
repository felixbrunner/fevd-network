"""
This module provides functionality to sample the largest companies from
CRSP data at the end of each month.

"""

import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from euraculus.data import DataMap


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
        end_date = sampling_date + relativedelta(months=self.back_offset)
        if end_date.month == 2 and end_date.day == 28:
            end_date += relativedelta(day=31)

        # load data
        df_back = self.data.load_crsp_data(start_date, sampling_date)
        df_forward = self.data.load_crsp_data(
            sampling_date + dt.timedelta(days=1), end_date
        )
        return (df_back, df_forward)

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
    def _get_last_sizes(df: pd.DataFrame) -> pd.Series:
        """Get last firm sizes from dataset.

        Create a series of last observed market capitalisation for
        all contained permnos.

        Args:
            df: CRSP data with column 'mcap'.

        Returns:
            last_size: Permno, size pairs with last observed sizes.

        """
        last_size = (
            df["mcap"]
            .unstack()
            .sort_index()
            .fillna(method="ffill", limit=1)
            .tail(1)
            .squeeze()
            .rename("last_size")
        )
        return last_size

    @staticmethod
    def _get_mean_sizes(df: pd.DataFrame) -> pd.Series:
        """Get average firm sizes from dataset.

        Return a series of average market capitalisations for contained permnos.

        Args:
            df : CRSP data with column 'mcap'.

        Returns:
            mean_size: Permno, size pairs with average observed sizes.

        """
        mean_size = df["mcap"].unstack().mean().squeeze().rename("mean_size")
        return mean_size

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
        self, df_back: pd.DataFrame, df_forward: pd.DataFrame
    ) -> pd.DataFrame:
        """Create summary statistics for the financial data provided.

        Return a summaring dataframe where the index consists
        of the permnos present in the dataframe.
        Columns are:
            ticker
            has_all_days
            has_obs
            last_size
            mean_size
            size_rank

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
            .join(self._get_last_sizes(df_back))
            .join(self._get_mean_sizes(df_back))
        )

        # set next obs to TRUE if there are no subsequent observations at all
        if not any(df_summary["has_next_obs"]):
            df_summary["has_next_obs"] = True

        # cross-sectional size rank
        df_summary["last_size_rank"] = (
            df_summary["last_size"]
            .where(df_summary["has_all_days"])
            .where(df_summary["has_next_obs"])
            .rank(ascending=False)
        )
        df_summary["mean_size_rank"] = (
            df_summary["mean_size"]
            .where(df_summary["has_all_days"])
            .where(df_summary["has_next_obs"])
            .rank(ascending=False)
        )

        return df_summary

    def _select_largest(self, df_summary: pd.DataFrame, method: str = "mean") -> list:
        """Pick N lagest companies.

        Return a list of permnos of the n companies
        with the highest market capitalisation end of year.

        Args:
            df_summary: Summary data to determine selection.
            method: Selection criterion, can be 'last' or 'mean' to select largest
                companies based on last or mean size.

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
            df_summary.sort_values("{}_size_rank".format(method))
            .head(self.n_assets)
            .index.get_level_values("permno")
            .tolist()
        )
        return permnos

    def sample(self, sampling_date: str) -> tuple:
        """Preprocess CRSP data to provide sample of large caps.

        Prepare asset return and variance datasets for a specified sampling date.

        Args:
           sampling_date: Sampling date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.

        Returns:
            df_historic: Sampled data of the period prior to the sampling date.
            df_future: Sampled data of the period after the sampling date.
            df_summary: Summary data used to determine selection.

        """
        # set up data
        sampling_date = self._prepare_date(sampling_date)
        df_historic, df_future = self._load_sampling_data(sampling_date)

        # select assets
        df_summary = self._describe_sampling_data(df_historic, df_future)
        permnos = self._select_largest(df_summary, method="mean")

        # slice
        df_historic = df_historic[
            df_historic.index.isin(
                pd.MultiIndex.from_product(
                    [
                        df_historic.index.get_level_values("date").unique().tolist(),
                        permnos,
                    ],
                    names=["date", "permno"],
                )
            )
        ]
        df_future = df_future[
            df_future.index.isin(
                pd.MultiIndex.from_product(
                    [
                        df_future.index.get_level_values("date").unique().tolist(),
                        permnos,
                    ],
                    names=["date", "permno"],
                )
            )
        ]

        return (df_historic, df_future, df_summary)
