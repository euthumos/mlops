if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

import pandas as pd

@custom
def transform_custom(df, *args, **kwargs):
    """
    df: DataFrame returned by read_yellow_raw
    """
    # Compute trip duration in minutes
    df['duration'] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    # Filter to trips between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Convert IDs to strings
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # Print the new size
    n2 = len(df)
    print(f'Prepared dataset size: {n2:,} records')

    return df
