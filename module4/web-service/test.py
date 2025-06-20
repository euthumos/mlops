#!/usr/bin/env python3
import argparse
import pandas as pd
import requests

def main(year: int, month: int, chunk_size: int = 5000):
    # 1. Build path based on year/month
    month_str = f"{month:02d}"
    path = f"../../datasets/yellow_tripdata_{year}-{month_str}.parquet"

    # 2. Only load needed columns
    COLUMNS = [
        'PULocationID',
        'DOLocationID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime'
    ]
    df = pd.read_parquet(path, columns=COLUMNS)

    # 3. Convert datetimes to ISO strings
    for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
        df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # 4. Send in chunks and accumulate weighted means
    total_rows   = len(df)
    chunk_means  = []
    chunk_counts = []
    url = 'http://127.0.0.1:9696/predict'

    for start in range(0, total_rows, chunk_size):
        end   = min(start + chunk_size, total_rows)
        chunk = df.iloc[start:end]
        payload = chunk.to_dict(orient='records')

        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        mean_duration = resp.json()['duration']

        chunk_means.append(mean_duration)
        chunk_counts.append(len(chunk))
        print(f"Chunk {start}-{end} → mean {mean_duration:.4f} min")

    # 5. Compute overall weighted mean
    overall = sum(m * c for m, c in zip(chunk_means, chunk_counts)) / sum(chunk_counts)
    print(f"\nProcessed {total_rows} rows in {len(chunk_means)} chunks.")
    print(f"Overall mean predicted duration: {overall:.4f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute mean trip duration predictions for a given year/month"
    )
    parser.add_argument(
        "-y", "--year",
        type=int,
        required=True,
        help="Year of the yellow_tripdata file, e.g. 2023"
    )
    parser.add_argument(
        "-m", "--month",
        type=int,
        choices=range(1, 13),
        required=True,
        help="Month (1–12) of the yellow_tripdata file"
    )
    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows to send per request (default: 5000)"
    )

    args = parser.parse_args()
    main(args.year, args.month, args.chunk_size)
