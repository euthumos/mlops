if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

import pandas as pd

@custom
def transform_custom(*args, **kwargs):
    # dataset were saved locally
    filename = '../datasets/yellow_tripdata_2023-03.parquet'
    df = pd.read_parquet(filename)
    # print the number of records loaded
    n = len(df)
    print(f'Loaded {n:,} records') 

    return df