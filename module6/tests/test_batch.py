import pytest
import pandas as pd
from datetime import datetime as dt

# import the functions to test
from batch import prepare_data


def make_datetime(hour, minute, second=0):
    return dt(2023, 1, 1, hour, minute, second)


def test_prepare_data_filters_and_formats():

    data = [
        (None, None, make_datetime(1, 1),   make_datetime(1, 10)),
        (1,    1,    make_datetime(2, 0),   make_datetime(2, 10)),
        (1,    None, make_datetime(2, 0),   make_datetime(2, 59)),
        (3,    4,    make_datetime(2, 0),   dt(2023,1,2,2,1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    result = prepare_data(df, categorical)

    # Expected: all rows have duration between 1 and 60 minutes inclusive
    # Row durations: 9, 10, 59, 24*60+1=1441 -> last should be filtered out (>60)
    assert all((result['duration'] >= 1) & (result['duration'] <= 60))
    assert len(result) == 3

    # Check that categorical columns are strings and no nulls remain
    for col in categorical:
        assert result[col].dtype == object
        assert result[col].isnull().sum() == 0

    # Check specific transformation: first row had None,None -> '-1','-1'
    first = result.iloc[0]
    assert first['PULocationID'] == '-1'
    assert first['DOLocationID'] == '-1'

    # Check duration values
    expected_durations = [9.0, 10.0, 59.0]
    assert pytest.approx(list(result['duration']), rel=1e-3) == expected_durations
    