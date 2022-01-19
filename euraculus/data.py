"""This module provides tools to download raw data from WRDS via SQL queries.

The WRDSDownloader object will establish a connection to the database and enable
some pre-defined queries to download the required data for the project.

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
        path (pathlib.Path): Path to the data storage directory.
        files (list): List of paths of all files in all subdirectories.

    """

    def __init__(self, datapath: str):
        """Sets up the datamap of local filesystem.

        Args:
            datapath (str): Path to the topmost local data folder named 'data'.

        """
        # path
        if datapath == "":
            self.datapath = Path().cwd() / "data"
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
            same_stem = [path.stem == f.stem for f in self.files]
            if sum(same_stem) == 1:
                hit = [file for file, same in zip(self.files, same_stem) if same][0]
                warnings.warn(
                    "file at '{}' does not exist, reading file '{}' from datamap instead".format(
                        path, hit
                    )
                )
                path = hit
            else:
                raise ValueError(
                    "file at '{}' does not exist, found {} similar files in datamap: {}".format(
                        path,
                        sum(same_stem),
                        [
                            str(file)
                            for file, same in zip(self.files, same_stem)
                            if same
                        ],
                    )
                )
            # except:
            #     raise ValueError("file at '{}' does not exist".format(path))
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
