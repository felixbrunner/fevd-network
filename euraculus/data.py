"""This module provides convenient access to the local filesystem for data.

"""

from pathlib import Path
import pandas as pd
import datetime as dt
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
        path = Path(path.strip("/"))
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

    def read(self, path: str) -> object:
        """Looks for file in datamap and loads it from disk.

        Args:
            path: Location where data is to be stored. Can be a full path,
                a filename, or a filename with extension.

        Returns:
            data: Python object loaded from disk.

        """

        # complete path
        path = Path(path.strip("/"))
        if not self.datapath in path.parents:
            path = self.datapath / path

        # check path variables
        if not path.exists():
            hits = self.search(path.name)
            if len(hits) == 1:
                warnings.warn(
                    "file at '{}' does not exist, reading file '{}' from datamap instead".format(
                        path, hits[0]
                    )
                )
                path = hits[0]
            else:
                raise ValueError(
                    "file at '{}' does not exist, found {} similar files in datamap: {}".format(
                        path,
                        len(hits),
                        hits if len(hits) > 0 else None,
                    )
                )
        extension = path.suffix

        # read csv
        if extension == ".csv":
            data = pd.read_csv(path)

        # read json
        elif extension == ".json":
            with open(path, "rb") as f:
                data = json.load(f)

        # read pickle
        elif extension in [".p", ".pkl", ".pickle"]:
            with open(path, "rb") as f:
                data = pickle.load(f)

        # handle other extensions
        else:
            raise NotImplementedError(
                "reading extension '{}' not implemented".format(extension)
            )

        return data

    def search(self, query: str) -> list:
        """Find filepaths in datamap corresponding to a search query.

        Args:
            query: Search term, can be filename or part of a path.

        Returns:
            hits: List of found filepaths.

        """
        querypath = Path(query)

        # search for exact filename matches
        same_name = [querypath.name == file.name for file in self.files]
        if sum(same_name) > 0:
            hits = [file for file, same in zip(self.files, same_name) if same]
            return hits

        # search for stem matches
        same_stem = [querypath.stem == file.stem for file in self.files]
        if sum(same_stem) > 0:
            hits = [file for file, same in zip(self.files, same_stem) if same]
            return hits

        # search for partial matches
        in_filepath = [query in str(file) for file in self.files]
        hits = [file for file, inpath in zip(self.files, in_filepath) if inpath]
        return hits

    def load_famafrench_factors(
        self, year: int = None, model: str = None
    ) -> pd.DataFrame:
        """Loads Fama/French factor data from drive.

        Contained series are:
            mktrf
            smb
            hml
            rf
            umd

        Args:
            year (optional): To only load observation of a single year.
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

        # slice out single year
        if year:
            df_famafrench = df_famafrench[df_famafrench.index.year == year]

        return df_famafrench

    def load_spy_data(self, year: int = None, series: str = None) -> pd.DataFrame:
        """Loads SPY data from disk.

        Args:
            year (int): Year to be loaded (optional), defaults to loading all years.
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

        # slice out single year
        if year:
            df_spy = df_spy[df_spy.index.year == year]

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
                    date
                )
            )

        return date

    def load_crsp_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Loads raw CRSP data for a given date range from disk.

        Args:
            start_date: First date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.
            end_date: Last date as dt.datetime or string, e.g. format 'YYYY-MM-DD'.

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
                df_year = self.read("/raw/crsp_{}.pkl".format(year))
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
                df_year = df_year.loc[
                    (df_year.index.get_level_values("date") >= start_date)
                ]
            if year == end_date.year:
                df_year = df_year.loc[
                    (df_year.index.get_level_values("date") <= end_date)
                ]

            # append output
            df_crsp = df_crsp.append(df_year)

        return df_crsp
