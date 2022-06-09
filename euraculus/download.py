"""This module provides tools to download raw data from WRDS via SQL queries.

The WRDSDownloader object will establish a connection to the database and enable
some pre-defined queries to download the required data for the project.

"""

import datetime as dt
import sys

import pandas as pd
import wrds
import yfinance


class WRDSDownloader:
    """Provides access to WRDS Database.

    Attributes:
        connection (wrds.Connection): Connection object to WRDS.

    """

    def __init__(self, connect: bool = True):
        """Sets up the Downloader to connect to WRDS.

        Will ask for WRDS credentials when connecting for the first time.

        Args:
            connect (bool):

        """
        # database connection
        if connect:
            self._connection = wrds.Connection()
        else:
            self._connection = connect

    @property
    def connection(self):
        """wrds.Connection: Connection to WRDS database with credentials."""
        if self._connection:
            return self._connection
        else:
            self._connection = wrds.Connection()
            return self._connection

    @connection.setter
    def connection(self):
        self._connection = wrds.Connection()

    def _close_connection(self):
        """Close the connection to the WRDS database."""
        self.connection.close()

    def _connect(self):
        """Make a connection to the WRDS database."""
        self.connection.connect()

    def _create_pgpass_file(self):
        """Create a .pgpass file to store WRDS connection credentials.

        Use the existing username and password if already connected to WRDS,
        or prompt for that information if not.

        The .pgpass file may contain connection entries for multiple databases,
        so we take care not to overwrite any existing entries unless they
        have the same hostname, port, and database name.

        On Windows, this file is actually called "pgpass.conf"
        and is stored in the %APPDATA%\postgresql directory.
        This must be handled differently.

        """
        self.connection.create_pgpass_file()

    def list_libraries(self) -> list:
        """Return all accessible WRDS libraries.

        Returns:
            libraries (list): List of available libraries.

        """
        libraries = self.connection.list_libraries()
        return libraries

    def list_tables(self, library: str = "crsp") -> list:
        """Return a list of all the views/tables within a library.

        Args:
            library (str): Name of the desired library.

        Returns:
            library_tables (str): List of available tables in the library.

        """
        library_tables = self.connection.list_tables(library=library)
        return library_tables

    def describe_table(
        self, library: str = "crsp", table: str = "dsf"
    ) -> pd.core.frame.DataFrame:
        """Describe all columns in a table.

        Args:
            library (str): Name of the desired library.
            table (str): Name of the desired table.

        Returns:
            table_description (pandas.DataFrame): Description of columns in the table.

        """
        table_description = self.connection.describe_table(library=library, table=table)
        return table_description

    def query(self, query: str) -> pd.core.frame.DataFrame:
        """Run a SQL query through the WRDS connection.

        Args:
            query (str): SQL query as a string.

        Returns:
            df (pandas.DataFrame): The downloaded data in tabular form.

        """
        t0 = dt.datetime.today()
        df = self.connection.raw_sql(query)
        t1 = dt.datetime.today()
        print(
            "collected {:.2f} MB at {} in {} seconds".format(
                sys.getsizeof(df) / 1e6, str(dt.datetime.today()), (t1 - t0).seconds
            )
        )
        return df

    def _create_crsp_year_query(self, year: int) -> str:
        """Create a SQL query string to download a single year.

        Args:
            year (int): Year to be downloaded.

        Returns:
            query (str): SQL query to download the respective year's data.

        """
        query = """
            SELECT 
            a.permno,
            b.ticker,
            a.date,
            a.ret,
            c.dlret,
            a.shrout * a.prc AS mcap,
            (0.3607) * POWER(LOG(NULLIF(a.askhi, 0)) - LOG(NULLIF(a.bidlo, 0)), 2) AS var,
            (0.3607) * POWER(LOG(NULLIF(a.ask, 0)) - LOG(NULLIF(a.bid, 0)), 2) AS noisevar
        
            FROM crsp.dsf AS a
        
            LEFT JOIN crsp.msenames AS b
            ON a.permno=b.permno
            AND b.namedt<=a.date
            AND a.date<=b.nameendt
        
            LEFT JOIN crsp.msedelist AS c
            ON a.permno=c.permno
            AND a.date=c.dlstdt
        
            WHERE a.date BETWEEN '01/01/{}' AND '12/31/{}'
            AND b.exchcd BETWEEN 1 AND 3
            AND b.shrcd BETWEEN 10 AND 11
            """.format(
            year, year
        )

        return query

    def download_crsp_year(self, year: int) -> pd.core.frame.DataFrame:
        """Download a single year of CRSP dsf data and make first adjustments.

        Args:
            year (int): Year to be downloaded.

        Returns:
            df (pandas.DataFrame): The downloaded and adjusted data in tabular form.

        """
        # SQL qumake_crsp_query
        query = self._create_crsp_year_query(year)
        df = self.query(query)

        # edit data formats
        df["date"] = pd.to_datetime(df.date, yearfirst=True)
        df["permno"] = df.permno.astype(int)

        # adjust returns for delisting
        df["retadj"] = (1 + df["ret"].fillna(0)) * (1 + df["dlret"].fillna(0)) - 1
        df["retadj"] = df["retadj"].where(df["ret"].notna() | df["dlret"].notna())
        df = df.drop(columns=["ret", "dlret"])

        # declare index & sort
        df.set_index(["date", "permno"], inplace=True)
        df = df.sort_index()

        return df

    def download_delisting_returns(self) -> pd.core.frame.DataFrame:
        """Download CRSP delisting returns.

        Returns:
            df (pandas.DataFrame): The downloaded data in tabular form.

        """
        query = """
            SELECT
            permno,
            dlret,
            dlstdt AS date
        
            FROM crsp.msedelist
            """
        df = self.query(query).set_index("permno")
        return df

    def download_stocknames(self) -> pd.core.frame.DataFrame:
        """Download CRSP company names and tickers from MSEnames.

        Downloaded columns are:
            comnam
            ticker
            namedt
            nameendt
            exchcd

        Args:
            connection (wrds.Connection): Connection object to CRSP.

        Returns:
            df (pandas.DataFrame): The downloaded data in tabular form.

        """
        query = """
            SELECT
            permno,
            comnam, 
            ticker,
            namedt,
            nameendt,
            exchcd
        
            FROM crsp.msenames
            """
        df = self.query(query)
        df = df.astype({"permno": int, "exchcd": int}).set_index("permno")
        return df

    def download_crsp_a_stocknames(self) -> pd.core.frame.DataFrame:
        """Download full stock names from CRSP annual stock.

        Note:
            Method currently not in use.

        Args:
            connection (wrds.Connection): Connection object to CRSP.

        Returns:
            df (pandas.DataFrame): The downloaded data in tabular form.
        """
        query = """
            SELECT
            permno,
            comnam, 
            ticker,
            st_date,
            end_date,
            exchcd
        
            FROM crsp_a_stock.stocknames
            """
        df = self.query(query)
        df = df.astype({"permno": int, "exchcd": int}).set_index("permno")
        return df

    def download_famafrench_factors(self) -> pd.core.frame.DataFrame:
        """Download Fama and French factor data from CRSP.

        Returns:
            df (pandas.DataFrame): The downloaded data in tabular form.

        """
        query = """
            SELECT *
            FROM ff_all.factors_daily 
            """
        #    WHERE date BETWEEN '01/01/2000' AND '12/31/2019'

        df = self.query(query).set_index("date")
        return df

    def download_spy_data(self) -> pd.core.frame.DataFrame:
        """Download SPY return and variance data from CRSP.

        Returns:
            df (pandas.DataFrame): The downloaded data in tabular form.

        """
        query = """
            SELECT
            a.date,
            a.prc,
            a.ret,
            (0.3607) * POWER(LOG(NULLIF(a.askhi, 0)) - LOG(NULLIF(a.bidlo, 0)), 2) AS var,
            (0.3607) * POWER(LOG(NULLIF(a.ask, 0)) - LOG(NULLIF(a.bid, 0)), 2) AS noisevar
        
            FROM crsp.dsf AS a
        
            LEFT JOIN crsp.msenames AS b
            ON a.permno=b.permno
        
            WHERE b.exchcd=4
            AND b.ticker='SPY'
            AND b.comnam='SPDR TRUST'
            """
        df = self.query(query).set_index("date")
        return df


def download_yahoo_data(ticker: str) -> pd.DataFrame:
    """Download historical data from Yahoo! Finance.

    Args:
        ticker: The ticker to download as displayed on the website.

    Returns:
        df: The downloaded and adjusted data in tabular form.

    """
    df = yfinance.Ticker(ticker).history(period="max")
    return df
