# import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from euraculus.data import DataMap

datamap = DataMap(datapath="/home/rubelrennfix/projects/euraculus/data")


class TestPrepareLogVariances:
    """This class serves to test various cases of preparing log variance data."""

    def test_full_column(self):
        df_var = pd.DataFrame(data=[[1], [1], [1]])
        df_noisevar = pd.DataFrame(data=[[2], [2], [2]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[1], [1], [1]]))
        assert_frame_equal(output, expected)

    def test_empty_column(self):
        _ = np.nan
        df_var = pd.DataFrame(data=[[_], [_], [_]])
        df_noisevar = pd.DataFrame(data=[[_], [_], [_]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[_], [_], [_]]))
        assert_frame_equal(output, expected)

    def test_zero_column(self):
        df_var = pd.DataFrame(data=[[0], [0], [0]])
        df_noisevar = pd.DataFrame(data=[[2], [2], [2]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[2], [2], [2]]))
        assert_frame_equal(output, expected)

    def test_middle_zero(self):
        df_var = pd.DataFrame(data=[[1], [0], [1]])
        df_noisevar = pd.DataFrame(data=[[2], [0], [2]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[1], [2], [1]]))
        assert_frame_equal(output, expected)

    def test_middle_empty(self):
        _ = np.nan
        df_var = pd.DataFrame(data=[[1], [_], [1]])
        df_noisevar = pd.DataFrame(data=[[2], [_], [2]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[1], [2], [1]]))
        assert_frame_equal(output, expected)

    def test_begin_zero(self):
        df_var = pd.DataFrame(data=[[0], [1], [1]])
        df_noisevar = pd.DataFrame(data=[[0], [2], [2]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[2], [1], [1]]))
        assert_frame_equal(output, expected)

    def test_begin_empty(self):
        _ = np.nan
        df_var = pd.DataFrame(data=[[_], [1], [1]])
        df_noisevar = pd.DataFrame(data=[[_], [2], [2]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[2], [1], [1]]))
        assert_frame_equal(output, expected)

    def test_end_zero(self):
        df_var = pd.DataFrame(data=[[1], [1], [0]])
        df_noisevar = pd.DataFrame(data=[[2], [2], [0]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[1], [1], [2]]))
        assert_frame_equal(output, expected)

    def test_end_empty(self):
        _ = np.nan
        df_var = pd.DataFrame(data=[[1], [1], [_]])
        df_noisevar = pd.DataFrame(data=[[2], [2], [_]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[1], [1], [_]]))
        assert_frame_equal(output, expected)

    def test_middle_zeros(self):
        df_var = pd.DataFrame(data=[[1], [0], [0], [1]])
        df_noisevar = pd.DataFrame(data=[[2], [0], [0], [2]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[1], [2], [2], [1]]))
        assert_frame_equal(output, expected)

    def test_middle_empties(self):
        _ = np.nan
        df_var = pd.DataFrame(data=[[1], [_], [_], [1]])
        df_noisevar = pd.DataFrame(data=[[2], [_], [_], [2]])
        output = datamap.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
        expected = np.log(pd.DataFrame(data=[[1], [2], [2], [1]]))
        assert_frame_equal(output, expected)
