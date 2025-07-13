import os
import sys
import boto3
import pandas as pd
from datetime import datetime
from batch import read_data_io, get_output_path

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def save_data(df: pd.DataFrame, path: str):
    """
    Save a DataFrame to S3 or local filesystem using the same
    storage_options that batch.py uses for read_parquet/write_parquet.
    """
    s3_endpoint = os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint:
        storage_opts = {
            "client_kwargs": {
                "endpoint_url": s3_endpoint
            }
        }
    else:
        storage_opts = None

    df.to_parquet(
        path,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=storage_opts
    )

if __name__ == "__main__":
    # Step 0: ensure LocalStack bucket exists
    s3_endpoint = os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint:
        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1"
        )
        bucket = "nyc-duration"
        try:
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket '{bucket}'")
        except s3.exceptions.BucketAlreadyOwnedByYou:
            pass

    # Step 1: build the test DataFrame for January 2023
    data = [
        (None, None, dt(1, 1),   dt(1, 10)),
        (1,    1,    dt(1, 2),   dt(1, 10)),
        (1,    None, dt(1, 2),   dt(1, 2, 59)),
        (3,    4,    dt(1, 2),   dt(2, 2, 1)),
    ]
    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime"
    ]
    df_input = pd.DataFrame(data, columns=columns)

    # Step 2: write input to S3
    input_path = os.getenv("INPUT_FILE_PATTERN").format(year=2023, month=1)
    save_data(df_input, input_path)
    print("Uploaded test data to:", input_path)

    # Step 3: run the batch job
    ret = os.system(f"{sys.executable} batch.py 2023 1")
    assert ret == 0, "batch.py failed"
    print("batch.py completed successfully")

    # Step 4: read and verify the results
    output_path = get_output_path(2023, 1)
    df_result = read_data_io(output_path)
    print("Result preview:")
    print(df_result.head())

    # Optionally, assert on the sum of predicted durations:
    total = df_result["predicted_duration"].sum()
    expected = 36.277250  # from the model run on this test data
    assert abs(total - expected) < 1e-3, f"Sum mismatch: got {total}, expected {expected}"
    print(f"Sum of predicted durations: {total:.6f} (as expected)")