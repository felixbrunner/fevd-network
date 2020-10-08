import pandas as pd
import numpy as np
import datetime as dt
import sys

import src


def query_SQL(connection, query):
    '''Runs a SQL query through a given connection.'''
    t0 = dt.datetime.today()
    df = connection.raw_sql(query)
    t1 = dt.datetime.today()
    print('collected {:.2f} MB on {} in {} seconds'\
          .format(sys.getsizeof(df)/1e+6,
                  str(dt.datetime.today()),
                  (t1-t0).seconds))
    return df


def create_crsp_query(year):
    '''Creates a SQL query string to download a single year.'''
    query = '''
        SELECT 
        a.permno,
        b.ticker,
        a.date,
        a.ret,
        c.dlret,
        a.shrout * a.prc AS mcap,
        SQRT(0.3607) * POWER(LOG(NULLIF(a.askhi, 0)) - LOG(NULLIF(a.bidlo, 0)), 1) AS vola
    
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
        '''\
        .format(year, year)
    
    return query
    
def download_crsp_year(connection, year):
    '''Downloads a single year of CRSP dsf data and makes
    first adjustments.
    '''
    # SQL qumake_crsp_query
    query = create_crsp_query(year)
    df = query_SQL(connection, query)
    
    # edit data formats
    df['date'] = pd.to_datetime(df.date, yearfirst=True)
    df['permno'] = df.permno.astype(int)
    
    # adjust returns for delisting
    df['retadj'] = (1+df['ret'].fillna(0))*(1+df['dlret'].fillna(0))-1
    df['retadj'] = df['retadj'].where(df['ret'].notna() | df['dlret'].notna())
    df = df.drop(columns=['ret', 'dlret'])
    
    # declare index & sort
    df.set_index(['date','permno'], inplace=True)
    df = df.sort_index()
    
    return df


def download_delisting(connection):
    '''Downloads CRSP delisting returns.'''
    query = '''
        SELECT
        permno,
        dlret,
        dlstdt AS date
    
        FROM crsp.msedelist
        '''
    df = query_SQL(connection, query).set_index('permno')
    return df


def download_descriptive(connection):
    '''Downloads CRSP descriptive data'''
    query = '''
        SELECT
        permno,
        comnam, 
        ticker,
        namedt,
        nameendt,
        exchcd
    
        FROM crsp.msenames
        '''
    df = query_SQL(connection, query)
    df = df.astype({'permno': int, 'exchcd': int})\
           .set_index('permno')
    return df


def download_descriptive_a_stocknames(connection):
    ''''''
    query = '''
        SELECT
        permno,
        comnam, 
        ticker,
        st_date,
        end_date,
        exchcd
    
        FROM crsp_a_stock.stocknames
        '''
    df = query_SQL(connection, query).set_index('permno')
    return df


def download_famafrench(connection):
    ''''''
    query = '''
        SELECT *
        FROM ff_all.factors_daily 
        '''
    #    WHERE date BETWEEN '01/01/2000' AND '12/31/2019'

    df = query_SQL(connection, query).set_index('date')
    return df


def download_SPY(connection):
    ''''''
    query = '''
        SELECT
        a.date,
        a.prc,
        a.ret,
        SQRT(0.361) * POWER(LOG(NULLIF(a.askhi, 0)) - LOG(NULLIF(a.bidlo, 0)), 1) AS vola
    
        FROM crsp.dsf AS a
    
        LEFT JOIN crsp.msenames AS b
        ON a.permno=b.permno
    
        WHERE b.exchcd=4
        AND b.ticker='SPY'
        AND b.comnam='SPDR TRUST'
        '''
    df = query_SQL(connection, query).set_index('date')
    return df