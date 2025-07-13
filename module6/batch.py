#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd


def get_input_path(year, month):
    default_input_pattern = "../datasets/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = "output/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data_io(filename):
    """
    Read the Parquet file and return a raw DataFrame (no transformations).
    If S3_ENDPOINT_URL is set, use it with storage_options for LocalStack.

    Parameters:
    - filename: path or URL to a Parquet file

    Returns:
    - Raw DataFrame
    """

    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }
        return pd.read_parquet(filename, storage_options=options)
    else:
        return pd.read_parquet(filename)


def prepare_data(df, categorical):
    """
    Take a raw DataFrame and apply all preprocessing steps:
    - compute trip duration
    - filter implausible durations
    - fill and convert categorical features

    Parameters:
    - df: raw Pandas DataFrame with columns tpep_pickup_datetime, tpep_dropoff_datetime
    - categorical: list of categorical column names to process

    Returns:
    - Cleaned DataFrame
    """

    df = df.copy()
    df['duration'] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df



def main(year, month):
    """
    Main entry point: load model, process data, predict durations,
    and write results to a Parquet file.

    Parameters:
    - year: int, four-digit year
    - month: int, month number (1-12)
    """

    # input_file = f"../datasets/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    # output_file = f"output/yellow_tripdata_{year:04d}-{month:02d}.parquet"

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df_raw = read_data_io(input_file)
    df = prepare_data(df_raw, categorical)

    df['ride_id'] = f"{year:04d}/{month:02d}_" + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    sum_pred = y_pred.sum()

    print(f"Sum of predicted durations for 2023-01: {sum_pred:.6f}")

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint_url:
        write_opts = {
            "client_kwargs": {
                "endpoint_url": s3_endpoint_url
            }
        }
    else:
        write_opts = None

    df_result.to_parquet(
        output_file,
        engine="pyarrow",
        index=False,
        storage_options=write_opts
    )

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)