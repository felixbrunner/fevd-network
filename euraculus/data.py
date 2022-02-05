"""This module provides convenient access to the local filesystem for data.

"""

from pathlib import Path
import pandas as pd
import pickle
import json
import warnings


class DataMap:
    """Serves to store and read data during the course of the project.

    DataMap serves to manage the local data storage on disk.

    Attributes:
        datapath (pathlib.Path): Path to the data storage directory.
        files (list): List of paths of all files in all subdirectories.

    """

    def __init__(self, datapath: str = None):
        """Set up the datamap of local filesystem.

        Args:
            datapath (str): Path to the topmost local data folder named 'data'.

        """
        # path
        if not datapath:
            self.datapath = Path().cwd() / "data"
            warnings.warn(
                "no datapath input, setting datapath to '{}'".format(self.datapath)
            )
        else:
            datapath = Path(datapath)
            if datapath.name != "data":
                raise ValueError("datapath must have name 'data'")
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

    def write(self, data: object, path: str):
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
                "no subdirectory selected, '{}' will be written to main data directory '{}'".format(
                    path.name, self.datapath
                )
            )
        elif not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("creating parent directory at '{}'".format(path.parent))
        if path.exists():
            warnings.warn("file at '{}' will be overwritten".format(path))

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
                "writing with extension '{}' not implemented".format(extension)
            )

        # save in files list
        self.files.append(path)
        print("file '{}' saved at '{}'".format(path.name, path.parent))

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
            # if len(hits) == 1:
            #     warnings.warn(
            #         "file at '{}' does not exist, reading file '{}' from datamap instead".format(
            #             path, hits[0]
            #         )
            #     )
            #     path = hits[0]
            # else:
            raise ValueError(
                "file at '{}' does not exist, found {} similar files in datamap: {}".format(
                    path,
                    len(hits),
                    hits if len(hits) > 0 else None,
                )
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
                "reading extension '{}' not implemented".format(path.suffix)
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

            # check for duplicates
            duplicates = set(df_left.columns).intersection(set(data.columns))
            if len(duplicates) > 0:
                warnings.warn(
                    "columns {} already stored and will be overwritten".format(
                        duplicates
                    )
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
            self.write(df_merged, path=path)
            del self.files[-1]

        # write data to disk if no file exists
        except ValueError:
            print("Creating file '{}' at {}.".format(path.name, path.parent))
            self.write(data, path=path)

    def load_famafrench_factors(
        self,
        model: str = None,
        # year: int = None
    ) -> pd.DataFrame:
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
                    "model '{}' not available, available options are ['capm', 'rf', 'ff3f', 'c4f']".format(
                        model
                    )
                )
            elif model == "capm":
                df_famafrench = df_famafrench[["mktrf"]]
            elif model == "rf":
                df_famafrench = df_famafrench[["rf"]]
            elif model == "ff3f":
                df_famafrench = df_famafrench[["mktrf", "smb", "hml"]]
            elif model == "c4f":
                df_famafrench = df_famafrench[["mktrf", "smb", "hml", "umd"]]

        # # slice out single year
        # if year:
        #     df_famafrench = df_famafrench[df_famafrench.index.year == year]

        return df_famafrench

    def load_spy_data(
        self,
        # year: int = None,
        series: str = None,
    ) -> pd.DataFrame:
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
                    "series '{}' not available, available options are {}".format(
                        series, list(df_spy.columns)
                    )
                )
            else:
                df_spy = df_spy[[series]]

        # # slice out single year
        # if year:
        #     df_spy = df_spy[df_spy.index.year == year]

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
                "date '{}' could not be converted to datetime, try format 'YYYY-MM-DD'".format(
                    dateinput
                )
            )

        return date

    @classmethod
    def _slice_daterange(
        cls, df: pd.DataFrame, start_date: str = None, end_date: str = None
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
                    "CRSP data for year {} does not exist locally, will be skipped".format(
                        year
                    )
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

    def load_sample(
        self, year: int, month: int, which: str = "back", column: str = None
    ) -> pd.DataFrame:
        """Loads monthly sampled CRSP data from disk.

        Args:
            year: Year of the sampling date.
            month: Month of the sampling date.
            which: Defines if forward or backward looking data should be loaded.
                Options are 'back' and 'forward'.
            column: Name of a single column to be loaded (optional).

        Returns:
            df: CRSP data in tabular form.
        """
        # load raw
        if which == "back":
            df = self.read("samples/{0}{1:0=2d}/df_back.csv".format(year, month))
        elif which == "forward":
            df = self.read("samples/{0}{1:0=2d}/df_forward.csv".format(year, month))

        # format
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "permno"])

        # return data matrix if column is chosen
        if column:
            df = df[column].unstack()

        return df.sort_index()

    def load_estimates(self, date: str, names: list = None) -> pd.DataFrame:
        """Read estimated values from disk.

        Args:
            date: Reference date to filter descriptive data.
                Can be dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            names: Names of the columns to be included in the output.

        Returns:
            df_estimates: The estimates read from disk.

        """
        # read and format
        date = self._prepare_date(date)
        df_estimates = self.read(
            path="samples/{0}{1:0=2d}/df_estimates.csv".format(date.year, date.month)
        )
        df_estimates = df_estimates.set_index("permno")

        # slice out particular estimates
        if names:
            if type(names) == str:
                names = [names]
            df_estimates = df_estimates[names]

        return df_estimates

    def make_sample_indices(self, year: int, month: int, which: str = "back"):
        """Collects excess returns data on three asset indices.

        Included indices are:
            - equally-weighted
            - value-weighted at the beginning of the observation period
            - the SPY index

        Args:
            year: Year of the sampling date.
            month: Month of the sampling date.
            which: Defines if forward or backward looking data should be loaded.
                Options are 'back' and 'forward'.

        Returns:
            df_indices: Index data in tabular form.

        """
        # load required data
        df_returns = self.load_sample(
            year=year, month=month, which=which, column="retadj"
        )
        df_mcaps = self.load_sample(year=year, month=month, which=which, column="mcap")
        df_rf = self.load_rf()
        df_spy = self.load_spy_data()[["ret"]]

        # prepare data
        df_returns -= df_rf.loc[df_returns.index].values
        weights = df_mcaps.iloc[0] / df_mcaps.iloc[0].sum()

        # outputs
        df_indices = pd.DataFrame(
            index=df_returns.index, columns=pd.Index(["ew", "vw", "spy"], name="index")
        )
        df_indices["ew"] = df_returns.mean(axis=1)
        df_indices["vw"] = (df_returns * weights).sum(axis=1)
        df_indices["spy"] = (
            df_spy.loc[df_returns.index].values - df_rf.loc[df_returns.index].values
        )

        return df_indices

    def lookup_ticker(self, tickers: list, date: str) -> pd.DataFrame:
        """Looks up information for a specific ticker on a given date.

        Args:
            ticker: The ticker to look up.
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
        ticker_data = df_descriptive[df_descriptive.ticker.isin(tickers)]
        return ticker_data
