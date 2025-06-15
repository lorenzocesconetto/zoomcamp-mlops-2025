#!/usr/bin/env python
# coding: utf-8

import pickle

import click
import numpy as np
import pandas as pd

with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


categorical = ["PULocationID", "DOLocationID"]


def read_data(year: int, month, type: str = "yellow"):
    df = pd.read_parquet(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"{type}_tripdata_{year:04d}-{month:02d}.parquet"
    )

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


@click.command()
@click.option("--year", required=True, type=int, help="Year of the trip data")
@click.option("--month", required=True, type=int, help="Month of the trip data")
def main(year, month):
    df = read_data(year, month)
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(f"Standard deviation of duration predictions: {np.std(y_pred):.3f}")
    print(f"Mean predicted duration: {np.mean(y_pred):.3f}")


if __name__ == "__main__":
    main()
