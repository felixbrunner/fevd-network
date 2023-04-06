"""This module provides convenient access to the local filesystem for data."""

import json
import math
import pickle
import warnings
from pathlib import Path
from collections import defaultdict
from dateutil.relativedelta import relativedelta

import pandas as pd

from euraculus.settings import DATA_DIR, TIME_STEP


class DataMap:
    """Serves to store and read data during the course of the project.

    DataMap serves to manage the local data storage on disk.

    Attributes:
        datapath (pathlib.Path): Path to the data storage directory.
        files (list): List of paths of all files in all subdirectories.
    """

    def __init__(self, datapath: Path = DATA_DIR):
        """Set up the datamap of local filesystem.

        Args:
            datapath (str): Path to the topmost local data folder named 'data'.
        """
        # path
        if datapath != DATA_DIR:
            warnings.warn(f"Initializing DataMap at non-standard datapath '{datapath}'")
        self.datapath = Path(datapath)

        # files
        self.files = []
        self._explore_path(self.datapath)

    def _explore_path(self, path: Path):
        """Appends all files in a given path to the files list.

        Args:
            path (pathlib.Path): Path object to the directory to be searched.
        """
        for subpath in path.iterdir():
            if subpath.is_dir():
                self._explore_path(subpath)
            else:
                self.files.append(subpath)

    def _refresh_map(self):
        """Updates the files attribute.

        Deletes all existing entries and searches the data directory and all
        subfolders for files to map.
        """
        self.files = []
        self._explore_path(self.datapath)

    def dump(self, data: object, path: str):
        """Save data on disk and extend map to new file.

        Args:
            data: Python object to be saved on disk and in datamap.
            path: Location where data is to be stored. Can be a full path,
                a filename, or a filename with extension.
        """
        # complete path
        path = Path(path)
        if not self.datapath in path.parents:
            path = self.datapath / path

        # check extension
        if path.suffix == "":
            if type(data) == pd.core.frame.DataFrame:
                path = path.with_suffix(".csv")
            else:
                path = path.with_suffix(".pickle")
        extension = path.suffix

        # prepare path variables
        if path.parent == self.datapath:
            warnings.warn(
                f"no subdirectory selected, '{path.name}' will be written to main data directory '{self.datapath}'"
            )
        elif not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"creating parent directory at '{path.parent}'")
        if path.exists():
            warnings.warn(f"file at '{path}' will be overwritten")

        # write as csv
        if extension == ".csv":
            data.to_csv(path)

        # write as json
        elif extension == ".json":
            with open(path, "wb") as f:
                json.dump(obj=data, file=f)

        # write as pickle
        elif extension in [".p", ".pkl", ".pickle"]:
            if type(data) == pd.DataFrame:
                data.to_pickle(path)
            else:
                path.touch()
                with open(path, "wb") as f:
                    pickle.dump(obj=data, file=f)

        # other file extensions
        else:
            raise NotImplementedError(
                f"writing with extension '{extension}' not implemented"
            )

        # save in files list
        self.files.append(path)
        print(f"file '{path.name}' saved at '{path.parent}'")

    def search(self, query: str, search_in: str = None) -> list:
        """Find filepaths in datamap corresponding to a search query.

        Args:
            query: Search term, can be filename or part of a path.

        Returns:
            hits: List of found filepaths.
        """
        querypath = Path(query)

        # search for exact filename matches
        same_name = [querypath.name == file.name for file in self.files]
        hits = [file for file, same in zip(self.files, same_name) if same]

        # search for stem matches
        if len(hits) == 0:
            same_stem = [querypath.stem == file.stem for file in self.files]
            hits = [file for file, same in zip(self.files, same_stem) if same]

        # search for partial matches
        if len(hits) == 0:
            in_filepath = [query in str(file) for file in self.files]
            hits = [file for file, inpath in zip(self.files, in_filepath) if inpath]

        if search_in:
            hits = [
                hit
                for hit in hits
                if search_in in [parent.name for parent in hit.parents]
            ]

        return hits

    def read(self, path: str) -> object:
        """Looks for file in datamap and loads it from disk.

        Args:
            path: Location where data is to be stored. Can be a full path,
                a filename, or a filename with extension.

        Returns:
            data: Python object loaded from disk.
        """
        # complete path
        path = Path(path)
        if not self.datapath in path.parents:
            path = self.datapath / path

        # check path variables
        if not path.exists():
            hits = self.search(query=path.name)
            raise ValueError(
                f"file at '{path}' does not exist, found {len(hits)} similar files in datamap: {hits if len(hits) > 0 else None}"
            )

        # read csv
        if path.suffix == ".csv":
            data = pd.read_csv(path)

        # read json
        elif path.suffix == ".json":
            with open(path, "rb") as f:
                data = json.load(f)

        # read pickle
        elif path.suffix in [".p", ".pkl", ".pickle"]:
            with open(path, "rb") as f:
                data = pickle.load(f)

        # handle other extensions
        else:
            raise NotImplementedError(
                f"reading extension '{path.suffix}' not implemented"
            )

        return data

    def store(self, data: pd.DataFrame, path: str):
        """Store data in an existing data file by extending it.

        Args:
            data: Data to be stored in tabular form.
            path: Location where data is to be stored. Can be a full path,
                a filename, or a filename with extension.
        """
        path = Path(path)

        # read exiting data from disk if it exists and combine
        try:
            df_left = self.read(path=path)
            if "date" in df_left.columns:
                df_left.date = pd.to_datetime(df_left.date)

            try:
                # check for duplicates
                duplicates = set(df_left.columns).intersection(set(data.columns))
                if len(duplicates) > 0:
                    warnings.warn(
                        f"columns {duplicates} already stored and will be overwritten"
                    )
                df_left = df_left.drop(columns=duplicates)

                # combine
                df_merged = (
                    df_left.merge(
                        data, how="outer", left_on=data.index.names, right_index=True
                    )
                    .set_index(data.index.names)
                    .dropna(how="all")
                )

                # write combined data to disk
                self.dump(df_merged, path=path)
                del self.files[-1]

            # extend series object
            except AttributeError:
                # check for duplicates
                s_old = df_left.set_index(df_left.columns[0]).squeeze()
                duplicates = set(s_old.index).intersection(set(data.index))
                if len(duplicates) > 0:
                    warnings.warn(
                        f"indices {duplicates} already stored and will be overwritten"
                    )

                # update
                s_merged = s_old.append(pd.Series(data))

                # write combined data to disk
                self.dump(s_merged, path=path)
                del self.files[-1]

        # write data to disk if no file exists
        except ValueError:
            print(f"creating file '{path.name}' at {path.parent}.")
            self.dump(data, path=path)

    def load_famafrench_factors(self, model: str = None) -> pd.DataFrame:
        """Loads Fama/French factor data from drive.

        Contained series are:
            mktrf
            smb
            hml
            rf
            umd

        Args:
            # year (optional): To only load observation of a single year.
            model (optional): To load data only for one of the following options
                ['capm', 'rf', 'ff3f', 'c4f']

        Returns:
            df_famafrench: Loaded data in tabular form.
        """

        # read & prepare
        df_famafrench = self.read("raw/ff_factors.pkl")
        df_famafrench.index = pd.to_datetime(df_famafrench.index, yearfirst=True)

        # subselection of factors
        if model:
            if not model in ["capm", "rf", "ff3f", "c4f"]:
                raise ValueError(
                    f"model '{model}' not available, available options are ['capm', 'rf', 'ff3f', 'c4f']"
                )
            elif model == "capm":
                df_famafrench = df_famafrench[["mktrf"]]
            elif model == "rf":
                df_famafrench = df_famafrench[["rf"]]
            elif model == "ff3f":
                df_famafrench = df_famafrench[["mktrf", "smb", "hml"]]
            elif model == "c4f":
                df_famafrench = df_famafrench[["mktrf", "smb", "hml", "umd"]]

        return df_famafrench

    def load_spy_data(self, series: str = None) -> pd.DataFrame:
        """Loads SPY data from disk.

        Args:
            # year (int): Year to be loaded (optional), defaults to loading all years.
            series (str): Column to be included in output.

        Returns:
            df (pandas.DataFrame): Selected SPY data in tabular format.
        """
        # read & prepare
        df_spy = self.read("raw/spy.pkl")
        df_spy.index = pd.to_datetime(df_spy.index, yearfirst=True)

        # subselection of factors
        if series:
            if not series in df_spy.columns:
                raise ValueError(
                    f"series '{series}' not available, available options are {list(df_spy.columns)}"
                )
            else:
                df_spy = df_spy[[series]]

        return df_spy

    def load_descriptive_data(self, date: str = None) -> pd.DataFrame:
        """Loads descriptive data from disk.

        Args:
            date (optional): Reference date to filter descriptive data.
                Can be dt.datetime or string, e.g. format 'YYYY-MM-DD'.

        Returns:
            df_descriptive: Descriptive data in tabular format.
        """
        # read & prepare
        df_descriptive = self.read("raw/descriptive.pkl")
        df_descriptive.index = df_descriptive.index.astype(int)
        df_descriptive = df_descriptive.astype({"exchcd": int})
        date_cols = ["namedt", "nameendt"]
        df_descriptive[date_cols] = df_descriptive[date_cols].apply(
            pd.to_datetime, format="%Y-%m-%d"
        )

        # filter at reference date
        if date:
            date = self._prepare_date(date)
            df_descriptive = df_descriptive.loc[
                (df_descriptive.namedt <= date) & (df_descriptive.nameendt >= date)
            ]

        return df_descriptive

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
                f"date '{dateinput}' could not be converted to datetime, try format 'YYYY-MM-DD'"
            )

        return date

    @classmethod
    def _slice_daterange(
        cls,
        df: pd.DataFrame,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Cuts out a slice between two dates from a dataframe.

        Args:
            df: A dataframe indexed by datetime.
            start_date: First date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            end_date: Last date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.

        Returns:
            df: A datetime slice of the input dataframe.
        """

        # slice beginning
        if start_date:
            start_date = cls._prepare_date(start_date)
            df = df.loc[(df.index.get_level_values("date") >= start_date)]

        # slice end
        if end_date:
            end_date = cls._prepare_date(end_date)
            df = df.loc[(df.index.get_level_values("date") <= end_date)]

        return df

    def load_crsp_data(
        self, start_date: str, end_date: str, column: str = None
    ) -> pd.DataFrame:
        """Loads raw CRSP data for a given date range from disk.

        Args:
            start_date: First date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            end_date: Last date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            column (optional): To select a single column and return in wide format.

        Returns:
            df_crsp: CRSP data in tabular format.
        """
        # set up
        start_date = self._prepare_date(start_date)
        end_date = self._prepare_date(end_date)
        df_crsp = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(arrays=[[], []], names=("date", "permno"))
        )

        # read and combine
        for year in range(start_date.year, end_date.year + 1):
            # read raw
            try:
                df_year = self.read("raw/crsp_{}.pkl".format(year))
            except ValueError:
                warnings.warn(
                    f"CRSP data for year {year} does not exist locally, will be skipped"
                )
                df_year = pd.DataFrame(
                    index=pd.MultiIndex.from_arrays(
                        arrays=[[], []], names=("date", "permno")
                    ),
                )

            # slice dates
            if year == start_date.year:
                df_year = self._slice_daterange(df=df_year, start_date=start_date)
            if year == end_date.year:
                df_year = self._slice_daterange(df=df_year, end_date=end_date)

            # append output
            df_crsp = df_crsp.append(df_year)

        # slice column and unstack
        if column:
            df_crsp = df_crsp[column].unstack()

        return df_crsp

    def load_rf(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Loads raw risk-free rate data for a given date range from disk.

        Args:
            start_date: First date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            end_date: Last date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.

        Returns:
            df_rf: Risk-free rate data in tabular format.
        """
        # read raw data
        df_rf = self.read("raw/ff_factors.pkl")[["rf"]]
        df_rf.index = pd.to_datetime(df_rf.index, yearfirst=True)

        # slice
        df_rf = self._slice_daterange(
            df=df_rf, start_date=start_date, end_date=end_date
        )

        return df_rf

    def load_historic(self, sampling_date: str, column: str = None) -> pd.DataFrame:
        """Load a sample of historic CRSP data from disk.

        Args:
            sampling_date: The sampling date as dt.datetime or string,
                e.g. format 'YYYY-MM-DD'.
            column: Name of a single column to be loaded (optional).

        Returns:
            df: Historic CRSP sample in tabular form.
        """
        # prepare & load raw
        sampling_date = self._prepare_date(sampling_date)
        df = self.read(f"samples/{sampling_date:%Y-%m-%d}/historic_daily.csv")

        # format
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "permno"])

        # return data matrix if column is chosen
        if column:
            if type(column) != str:
                raise ValueError(
                    f"specify single column as a string, not {type(column)}"
                )
            order = df.index.get_level_values("permno").unique().tolist()
            df = df[column].unstack().loc[:, order]

        return df

    def load_future(self, sampling_date: str, column: str = None) -> pd.DataFrame:
        """Load a sample of forward looking CRSP data from disk.

        Args:
            sampling_date: The sampling date as dt.datetime or string,
                e.g. format 'YYYY-MM-DD'.
            column: Name of a single column to be loaded (optional).

        Returns:
            df: Forward looking CRSP sample in tabular form.
        """
        # prepare & load raw
        sampling_date = self._prepare_date(sampling_date)
        df = self.read(f"samples/{sampling_date:%Y-%m-%d}/future_daily.csv")

        # format
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "permno"])

        # return data matrix if column is chosen
        if column:
            if type(column) != str:
                raise ValueError(
                    f"specify single column as a string, not {type(column)}"
                )
            order = df.index.get_level_values("permno").unique().tolist()
            df = df[column].unstack().loc[:, order]

        return df

    def load_historic_aggregates(
        self, sampling_date: str, column: str = None
    ) -> pd.DataFrame:
        """Load historic aggregates for a sampling period.

        Args:
            sampling_date: The sampling date as dt.datetime or string,
                e.g. format 'YYYY-MM-DD'.
            column: Name of a single column to be loaded (optional).

        Returns:
            df: Historic aggregate data in tabular form.
        """
        # prepare & load raw
        sampling_date = self._prepare_date(sampling_date)
        df = self.read(f"samples/{sampling_date:%Y-%m-%d}/historic_aggregates.csv")

        # format
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # return data matrix if column is chosen
        if column:
            if type(column) != str:
                raise ValueError(
                    f"specify single column as a string, not {type(column)}"
                )
            df = df[column]

        return df

    def load_future_aggregates(
        self, sampling_date: str, column: str = None
    ) -> pd.DataFrame:
        """Load future aggregates for a sampling period.

        Args:
            sampling_date: The sampling date as dt.datetime or string,
                e.g. format 'YYYY-MM-DD'.
            column: Name of a single column to be loaded (optional).

        Returns:
            df: Future aggregate data in tabular form.
        """
        # prepare & load raw
        sampling_date = self._prepare_date(sampling_date)
        df = self.read(f"samples/{sampling_date:%Y-%m-%d}/future_aggregates.csv")

        # format
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # return data matrix if column is chosen
        if column:
            if type(column) != str:
                raise ValueError(
                    f"specify single column as a string, not {type(column)}"
                )
            df = df[column]

        return df

    def load_asset_estimates(
        self,
        sampling_date: str = None,
        columns: list = None,
    ) -> pd.DataFrame:
        """Read estimated values on asset level from disk.

        Args:
            sampling_date (optional): Reference date to filter descriptive data.
                Can be dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            columns (optional): Names of the columns to be included in the output.

        Returns:
            df_estimates: The estimates read from disk.
        """
        # read single file
        if sampling_date:
            sampling_date = self._prepare_date(sampling_date)
            df_estimates = self.read(
                path=f"samples/{sampling_date:%Y-%m-%d}/asset_estimates.csv"
            )
            df_estimates = df_estimates.set_index("permno")

        # read multiple files
        else:
            df_estimates = pd.DataFrame()
            for sample in (self.datapath / "samples").iterdir():
                if sample.name == ".ipynb_checkpoints":
                    pass
                else:
                    try:
                        df_sample = self.read(
                            path=f"samples/{sample.name}/asset_estimates.csv"
                        )
                        df_sample["sampling_date"] = pd.to_datetime(sample.name)
                        df_estimates = df_estimates.append(df_sample)
                    except ValueError:
                        warnings.warn(
                            f"file at 'samples/{sample.name}/asset_estimates.csv' does not exist, will be skipped"
                        )
            df_estimates = df_estimates.set_index(["sampling_date", "permno"])

        # slice out particular estimates
        if columns:
            if type(columns) == str:
                columns = [columns]
            df_estimates = df_estimates[columns]

        return df_estimates

    def load_index_estimates(
        self,
        sampling_date: str = None,
        columns: list = None,
    ) -> pd.DataFrame:
        """Read estimated values on index level from disk.

        Args:
            sampling_date (optional): Reference date to filter descriptive data.
                Can be dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            columns (optional): Names of the columns to be included in the output.

        Returns:
            df_estimates: The estimates read from disk.
        """
        # read single file
        if sampling_date:
            sampling_date = self._prepare_date(sampling_date)
            df_estimates = self.read(
                path=f"samples/{sampling_date:%Y-%m-%d}/index_estimates.csv"
            )
            df_estimates = df_estimates.set_index("index")

        # read multiple files
        else:
            df_estimates = pd.DataFrame()
            for sample in (self.datapath / "samples").iterdir():
                if sample.name == ".ipynb_checkpoints":
                    pass
                else:
                    try:
                        df_sample = self.read(
                            path=f"samples/{sample.name}/index_estimates.csv"
                        )
                        df_sample["sampling_date"] = pd.to_datetime(sample.name)
                        df_estimates = df_estimates.append(df_sample)
                    except ValueError:
                        warnings.warn(
                            f"file at 'samples/{sample.name}/index_estimates.csv' does not exist, will be skipped"
                        )
            df_estimates = df_estimates.set_index(["sampling_date", "index"])

        # slice out particular estimates
        if columns:
            if type(columns) == str:
                columns = [columns]
            df_estimates = df_estimates[columns]

        return df_estimates

    def load_selection_summary(
        self,
        sampling_date: str = None,
        columns: list = None,
    ) -> pd.DataFrame:
        """Read summary table used for sample selection from disk.

        Args:
            sampling_date (optional): Reference date to filter descriptive data.
                Can be dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            columns (optional): Names of the columns to be included in the output.

        Returns:
            df_summary: The estimates read from disk.
        """
        # read single file
        if sampling_date:
            sampling_date = self._prepare_date(sampling_date)
            df_summary = self.read(
                path=f"samples/{sampling_date:%Y-%m-%d}/selection_summary.csv"
            )
            df_summary = df_summary.set_index("permno")

        # read multiple files
        else:
            df_summary = pd.DataFrame()
            for sample in (self.datapath / "samples").iterdir():
                if sample.name == ".ipynb_checkpoints":
                    pass
                else:
                    try:
                        df_sample = self.read(
                            path=f"samples/{sample.name}/selection_summary.csv"
                        )
                        df_sample["sampling_date"] = pd.to_datetime(sample.name)
                        df_summary = df_summary.append(df_sample)
                    except ValueError:
                        warnings.warn(
                            f"file at 'samples/{sample.name}/selection_summary.csv' does not exist, will be skipped"
                        )
            df_summary = df_summary.set_index(["sampling_date", "permno"])

        # slice out particular column
        if columns:
            if type(columns) == str:
                columns = [columns]
            df_summary = df_summary[columns]

        return df_summary

    def load_estimation_summary(
        self,
        sampling_date: str = None,
        columns: list = None,
        filename: str = "estimation_stats",
    ) -> pd.DataFrame:
        """Read summary table from estimation from disk.

        Args:
            sampling_date (optional): Reference date to filter descriptive data.
                Can be dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            columns (optional): Names of the columns to be included in the output.

        Returns:
            df_summary: The estimates read from disk.
        """
        # read single file
        if sampling_date:
            sampling_date = self._prepare_date(sampling_date)
            df_summary = self.read(
                path=f"samples/{sampling_date:%Y-%m-%d}/{filename}.csv"
            )
            df_summary = df_summary.set_index("statistic")

        # read multiple files
        else:
            df_summary = pd.DataFrame()
            for sample in (self.datapath / "samples").iterdir():
                if sample.name == ".ipynb_checkpoints":
                    pass
                else:
                    try:
                        df_sample = self.read(
                            path=f"samples/{sample.name}/{filename}.csv"
                        )
                        df_sample = df_sample.set_index("statistic")
                        df_sample.loc["sampling_date"] = pd.to_datetime(sample.name)
                        df_summary = df_summary.append(df_sample.T)
                    except ValueError:
                        warnings.warn(
                            f"file at 'samples/{sample.name}/{filename}.csv' does not exist, will be skipped"
                        )
            df_summary = df_summary.set_index("sampling_date").convert_dtypes()

        # slice out particular column
        if columns:
            if type(columns) == str:
                columns = [columns]
            df_summary = df_summary[columns]

        return df_summary

    def lookup_ticker(self, tickers: list, date: str) -> pd.DataFrame:
        """Looks up information for a specific ticker on a given date.

        Args:
            tickers: The tickers to look up.
            date: Reference date to filter descriptive data.
                Can be dt.datetime or string, e.g. format 'YYYY-MM-DD'.

        Returns:
            ticker_data: Descriptive data for the specific ticker.
        """
        # make ticker list
        if type(tickers) == str:
            tickers = tickers.split(",")

        # load & lookup
        df_descriptive = self.load_descriptive_data(date=date)
        ticker_data = pd.DataFrame(columns=df_descriptive.columns)
        for ticker in tickers:
            ticker_data = ticker_data.append(
                df_descriptive.loc[df_descriptive.ticker == ticker, :]
            )
        return ticker_data

    def lookup_permnos(self, permnos: list, date: str) -> pd.DataFrame:
        """Looks up information for a specific ticker on a given date.

        Args:
            permnos: The permnos to look up.
            date: Reference date to filter descriptive data.
                Can be dt.datetime or string, e.g. format 'YYYY-MM-DD'.

        Returns:
            ticker_data: Descriptive data for the specific ticker.
        """
        # make permno list
        if type(permnos) == int:
            permnos = [permnos]

        # load & lookup
        df_descriptive = self.load_descriptive_data(date=date)
        try:
            permno_data = df_descriptive.loc[permnos]
        except KeyError:
            permno_data = pd.DataFrame(columns=df_descriptive.columns)
            for permno in permnos:
                permno_data = permno_data.append(
                    df_descriptive.loc[df_descriptive.index == permno, :]
                )
        return permno_data

    def load_yahoo(self, ticker: str) -> pd.DataFrame:
        """Loads yahoo! Finance closing prices from disk.

        Args:
            ticker: Name of the raw data file.

        Returns:
            df: Selected price data in tabular format.
        """
        df = self.read(f"raw/{ticker}.pkl")
        return df

    def load_sic_table(self) -> pd.DataFrame:
        """Load SIC sector code table from disk.

        Returns:
            df: Table with SIC code translations.
        """
        df = self.read("raw/sic.pkl")
        return df

    def load_naics_table(self) -> pd.DataFrame:
        """Load NAICS sector code table from disk.

        Returns:
            df: Table with NAICS code translations.
        """
        df = self.read("raw/naics.pkl")
        return df

    def load_gics_table(self) -> pd.DataFrame:
        """Load GICS sector code table from disk.

        Returns:
            df: Table with GICS code translations.
        """
        df = self.read("raw/gics.pkl")
        return df

    def lookup_sic_divisions(self, codes: list) -> list:
        """Map a list of SIC codes into the division strings.

        Args:
            codes: A list-like of SIC integer codes.

        Returns:
            divisions: A list of strings with corresponding divisions.
        """
        # create lookup dictionary
        sic_divisions = [
            (range(100, 1000), "Agriculture, Forestry and Fishing"),
            (range(1000, 1500), "Mining"),
            (range(1500, 1800), "Construction"),
            (range(2000, 4000), "Manufacturing"),
            (
                range(4000, 5000),
                "Transportation, Communications, Electric, Gas and Sanitary service",
            ),
            (range(5000, 5200), "Wholesale Trade"),
            (range(5200, 6000), "Retail Trade"),
            (range(6000, 6800), "Finance, Insurance and Real Estate"),
            (range(7000, 9000), "Services"),
            (range(9100, 9730), "Public Administration"),
            (range(9900, 10000), "Nonclassifiable"),
        ]
        sic_names = defaultdict(lambda: "N/A")
        for code_range, division in sic_divisions:
            sic_names.update(dict.fromkeys(code_range, division))

        # lookup inputs
        divisions = [sic_names[code] for code in codes]
        return divisions

    def lookup_gics_sectors(self, codes: list) -> list:
        """Map a list of GICS codes into the sector strings.

        Args:
            codes: A list-like of GICS integer codes.

        Returns:
            divisions: A list of strings with corresponding divisions.
        """
        # create lookup dictionary
        gics = self.load_gics_table()
        gics_sectors = defaultdict(lambda: "N/A")
        gics_sectors.update(gics[gics["gictype"] == "GSECTOR"]["gicdesc"].to_dict())

        # lookup inputs
        sector_codes = [
            int(str(code)[:2]) if str(code).isnumeric() else math.nan for code in codes
        ]
        sectors = [gics_sectors[code] for code in sector_codes]
        return sectors

    def lookup_famafrench_sectors(
        self, codes: list, return_tickers: bool = False
    ) -> list:
        """Map a list of SIC codes into Fama/French's 12 sector definition.

        Mapping is taken from:
        http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html

        Args:
            codes: A list-like of SIC integer codes.
            return_tickers: Request tickers instead of full names.

        Returns:
            sectors: A list of strings with corresponding sectors.
        """
        # create lookup dictionary
        ff_sectors = [
            (
                range(100, 1000),
                "Consumer Nondurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys",
                "NoDur",
            ),
            (
                range(2000, 2400),
                "Consumer Nondurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys",
                "NoDur",
            ),
            (
                range(2700, 2750),
                "Consumer Nondurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys",
                "NoDur",
            ),
            (
                range(2770, 2800),
                "Consumer Nondurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys",
                "NoDur",
            ),
            (
                range(3100, 3200),
                "Consumer Nondurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys",
                "NoDur",
            ),
            (
                range(3940, 3990),
                "Consumer Nondurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys",
                "NoDur",
            ),
            (
                range(2500, 2520),
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                range(2590, 2600),
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                range(3630, 3660),
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                range(3710, 3712),
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                [3710, 3711, 3714, 3716],
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                [3750, 3751],
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                [3792],
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                range(3900, 3940),
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                range(3990, 4000),
                "Consumer Durables -- Cars, TVs, Furniture, Household Appliances",
                "Durbl",
            ),
            (
                range(2520, 2590),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(2600, 2700),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(2750, 2770),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3000, 3100),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3200, 3570),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3580, 3630),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3700, 3710),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                [3712, 3713, 3715],
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3717, 3750),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3752, 3792),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3793, 3800),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3830, 3840),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (
                range(3860, 3900),
                "Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing",
                "Manuf",
            ),
            (range(1200, 1400), "Oil, Gas, and Coal Extraction and Products", "Enrgy"),
            (range(2900, 3000), "Oil, Gas, and Coal Extraction and Products", "Enrgy"),
            (range(2800, 2830), "Chemicals and Allied Products", "Chems"),
            (range(2840, 2900), "Chemicals and Allied Products", "Chems"),
            (
                range(3570, 3580),
                "Business Equipment -- Computers, Software, and Electronic Equipment",
                "BusEq",
            ),
            (
                range(3660, 3693),
                "Business Equipment -- Computers, Software, and Electronic Equipment",
                "BusEq",
            ),
            (
                range(3694, 3700),
                "Business Equipment -- Computers, Software, and Electronic Equipment",
                "BusEq",
            ),
            (
                range(3810, 3830),
                "Business Equipment -- Computers, Software, and Electronic Equipment",
                "BusEq",
            ),
            (
                range(7370, 7380),
                "Business Equipment -- Computers, Software, and Electronic Equipment",
                "BusEq",
            ),
            (range(4800, 4900), "Telephone and Television Transmission", "Telcm"),
            (range(4900, 4950), "Utilities", "Utils"),
            (
                range(5000, 6000),
                "Wholesale, Retail, and Some Services (Laundries, Repair Shops)",
                "Shops",
            ),
            (
                range(7200, 7300),
                "Wholesale, Retail, and Some Services (Laundries, Repair Shops)",
                "Shops",
            ),
            (
                range(7600, 7700),
                "Wholesale, Retail, and Some Services (Laundries, Repair Shops)",
                "Shops",
            ),
            (range(2830, 2840), "Healthcare, Medical Equipment, and Drugs", "Hlth"),
            ([3693], "Healthcare, Medical Equipment, and Drugs", "Hlth"),
            (range(3840, 3860), "Healthcare, Medical Equipment, and Drugs", "Hlth"),
            (range(8000, 8100), "Healthcare, Medical Equipment, and Drugs", "Hlth"),
            (range(6000, 7000), "Finance", "Money"),
        ]
        ff_names = defaultdict(
            lambda: "Other -- Mines, Constr, BldMt, Trans, Hotels, Bus Serv, Entertainment"
        )
        ff_tickers = defaultdict(lambda: "Other")
        for code_range, name, ticker in ff_sectors:
            ff_names.update(dict.fromkeys(code_range, name))
            ff_tickers.update(dict.fromkeys(code_range, ticker))

        # lookup inputs
        if return_tickers:
            sectors = [ff_tickers[code] for code in codes]
        else:
            sectors = [ff_names[code] for code in codes]
        return sectors

    def load_nonoverlapping_historic(self, columns: list = None):
        """Load a dataframe with non-overlapping historic daily observations.

        Args:
            column: Name of the columns to load (optional).

        Returns:
            df_historic: Dataframe with historic observations.
        """
        df_historic = pd.DataFrame()

        for sample_path in (self.datapath / "samples").iterdir():
            sampling_date = self._prepare_date(sample_path.name)
            df_sample = self.load_historic(sampling_date=sampling_date)
            df_sample = self._slice_daterange(
                df=df_sample,
                start_date=sampling_date - TIME_STEP + relativedelta(days=1),
                end_date=sampling_date,
            )

            if columns is not None:
                df_sample = df_sample[columns]
            df_sample["sampling_date"] = sampling_date
            df_historic = df_historic.append(df_sample)

        return df_historic

    def load_nonoverlapping_future(self, columns: list = None):
        """Load a dataframe with non-overlapping future daily observations.

        Args:
            column: Name of the columns to load (optional).

        Returns:
            df_future: Dataframe with historic observations.
        """
        df_future = pd.DataFrame()

        for sample_path in (self.datapath / "samples").iterdir():
            sampling_date = self._prepare_date(sample_path.name)
            df_sample = self.load_future(sampling_date=sampling_date)
            df_sample = self._slice_daterange(
                df=df_sample,
                start_date=sampling_date + relativedelta(days=1),
                end_date=sampling_date + TIME_STEP,
            )

            if columns is not None:
                df_sample = df_sample[columns]
            df_sample["sampling_date"] = sampling_date
            df_future = df_future.append(df_sample)

        return df_future
