"""
This module provides a set of functions to required download data from CRSP.

"""

import pandas as pd
import datetime as dt
import sys


def query_SQL(connection, query: str) -> pd.DataFrame:
    """Run a SQL query through a given connection.

    Args:
        connection (wrds.Connection): Connection object to CRSP.
        query (str): SQL query as a string.

    Returns:
        df (pandas.DataFrame): The downloaded data in tabular form.

    """
    t0 = dt.datetime.today()
    df = connection.raw_sql(query)
    t1 = dt.datetime.today()
    print(
        "collected {:.2f} MB on {} in {} seconds".format(
            sys.getsizeof(df) / 1e6, str(dt.datetime.today()), (t1 - t0).seconds
        )
    )
    return df


def create_crsp_query(year: int) -> str:
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


def download_crsp_year(connection, year: int) -> pd.DataFrame:
    """Download a single year of CRSP dsf data and make first adjustments.

    Args:
        connection (wrds.Connection): Connection object to CRSP.
        year (int): Year to be downloaded.

    Returns:
        df (pandas.DataFrame): The downloaded and adjusted data in tabular form.

    """
    # SQL qumake_crsp_query
    query = create_crsp_query(year)
    df = query_SQL(connection, query)

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


def download_delisting(connection):
    """Download CRSP delisting returns.

    Args:
        connection (wrds.Connection): Connection object to CRSP.

    Returns:
        df (pandas.DataFrame): The downloaded and adjusted data in tabular form.

    """
    query = """
        SELECT
        permno,
        dlret,
        dlstdt AS date
    
        FROM crsp.msedelist
        """
    df = query_SQL(connection, query).set_index("permno")
    return df


def download_descriptive(connection):
    """Download CRSP descriptive data.

    Args:
        connection (wrds.Connection): Connection object to CRSP.

    Returns:
        df (pandas.DataFrame): The downloaded and adjusted data in tabular form.

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
    df = query_SQL(connection, query)
    df = df.astype({"permno": int, "exchcd": int}).set_index("permno")
    return df


def download_descriptive_a_stocknames(connection):
    """Download full stock names from CRSP.

    Args:
        connection (wrds.Connection): Connection object to CRSP.

    Returns:
        df (pandas.DataFrame): The downloaded and adjusted data in tabular form.
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
    df = query_SQL(connection, query).set_index("permno")
    return df


def download_famafrench(connection):
    """Download Fama and French factor data from CRSP.

    Args:
        connection (wrds.Connection): Connection object to CRSP.

    Returns:
        df (pandas.DataFrame): The downloaded and adjusted data in tabular form.

    """
    query = """
        SELECT *
        FROM ff_all.factors_daily 
        """
    #    WHERE date BETWEEN '01/01/2000' AND '12/31/2019'

    df = query_SQL(connection, query).set_index("date")
    return df


def download_SPY(connection):
    """Download SPY return and variance data from CRSP.

    Args:
        connection (wrds.Connection): Connection object to CRSP.

    Returns:
        df (pandas.DataFrame): The downloaded and adjusted data in tabular form.

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
    df = query_SQL(connection, query).set_index("date")
    return df
