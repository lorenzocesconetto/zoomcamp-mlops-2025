import pickle
from pathlib import Path
from typing import TypedDict

import mlflow
import pandas as pd
import xgboost as xgb
from airflow.sdk import Param, dag, task
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from typing_extensions import NotRequired

mlflow.set_experiment("nyc-taxi-experiment")

for path in ["models", "data", "mlruns", "artifacts"]:
    folder = Path(path)
    folder.mkdir(exist_ok=True)


class DataFrameMetadata(TypedDict):
    data_path: str
    record_count: int
    year: int
    month: int
    dict_vectorizer_path: NotRequired[str | None]


@dag(
    dag_display_name="Train Regressor for NYC Taxi durations",
    tags=["model_training"],
    schedule=None,
    catchup=False,
    params={
        "year": Param(2023, type="integer", minimum=2009, maximum=2030),
        "month": Param(3, type="integer", minimum=1, maximum=12),
        "model_type": Param("linear_regression", enum=["xgboost", "linear_regression"]),
    },
)
def train_regressor_nyc_taxi():
    """
    ### Train ML Regression Model
    This pipeline trains either an XGBoost or Linear Regression model to predict taxi trip
    durations in NYC.
    The process is tracked by MLFlow.
    """

    @task(multiple_outputs=True)
    def get_dates(*, params):
        """Extract year and month from params and calculate validation dates"""
        year = params["year"]
        month = params["month"]

        next_year = year if month < 12 else year + 1
        next_month = month + 1 if month < 12 else 1

        return {
            "train_year": year,
            "train_month": month,
            "val_year": next_year,
            "val_month": next_month,
        }

    @task()
    def read_dataframe(year: int, month: int) -> DataFrameMetadata:
        url = (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"yellow_tripdata_{year}-{month:02d}.parquet"
        )
        df = pd.read_parquet(url)

        print(f"{year=} {month=} total records: {len(df)}")

        df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ["PULocationID", "DOLocationID"]
        df[categorical] = df[categorical].astype(str)

        df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

        print(f"{year=} {month=} total records after filtering: {len(df)}")

        output_path = f"./data/{year}_{month:02d}.parquet"
        df.to_parquet(output_path)

        return {
            "data_path": output_path,
            "record_count": len(df),
            "year": year,
            "month": month,
        }

    @task()
    def dict_vectorize(meta: DataFrameMetadata, dv_path: str = None) -> DataFrameMetadata:
        df = pd.read_parquet(meta["data_path"])

        categorical = ["PU_DO"]
        numerical = ["trip_distance"]
        dicts = df[categorical + numerical].to_dict(orient="records")

        if dv_path is None:
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)
            dict_vectorizer_output_path = (
                f"./models/dict_vectorizer_{meta['year']}_{meta['month']:02d}.pkl"
            )
            with open(dict_vectorizer_output_path, "wb") as f:
                pickle.dump(dv, f)
        else:
            with open(dv_path, "rb") as f:
                dv = pickle.load(f)
            X = dv.transform(dicts)
            dict_vectorizer_output_path = dv_path

        # Save sparse matrix using scipy's sparse matrix format
        features_output_path = f"./data/features_{meta['year']}_{meta['month']:02d}.npz"
        sparse.save_npz(features_output_path, X)

        return {
            "data_path": features_output_path,
            "dict_vectorizer_path": dict_vectorizer_output_path,
            "record_count": meta["record_count"],
            "year": meta["year"],
            "month": meta["month"],
        }

    @task()
    def extract_targets(meta: DataFrameMetadata) -> str:
        """Extract target values and save to file"""

        df = pd.read_parquet(meta["data_path"])

        target = "duration"
        y = df[target].values

        # Save target values to file
        target_path = f"./data/target_{meta['year']}_{meta['month']:02d}.pkl"
        with open(target_path, "wb") as f:
            pickle.dump(y, f)

        return target_path

    @task()
    def train_model(
        train_features: DataFrameMetadata,
        val_features: DataFrameMetadata,
        train_target_path: str,
        val_target_path: str,
        *,
        params,
    ):
        # Load sparse feature matrices
        X_train = sparse.load_npz(train_features["data_path"])
        X_val = sparse.load_npz(val_features["data_path"])

        # Load target arrays from saved files
        with open(train_target_path, "rb") as f:
            y_train = pickle.load(f)
        with open(val_target_path, "rb") as f:
            y_val = pickle.load(f)

        model_type = params["model_type"]

        with mlflow.start_run() as run:
            mlflow.log_param("model_type", model_type)
            mlflow.log_artifact(
                train_features["dict_vectorizer_path"], artifact_path="preprocessor"
            )

            if model_type == "xgboost":
                # XGBoost training
                train = xgb.DMatrix(X_train, label=y_train)
                valid = xgb.DMatrix(X_val, label=y_val)

                best_params = {
                    "learning_rate": 0.09585355369315604,
                    "max_depth": 30,
                    "min_child_weight": 1.060597050922164,
                    "objective": "reg:linear",
                    "reg_alpha": 0.018060244040060163,
                    "reg_lambda": 0.011658731377413597,
                    "seed": 42,
                }

                mlflow.log_params(best_params)

                booster = xgb.train(
                    params=best_params,
                    dtrain=train,
                    num_boost_round=30,
                    evals=[(valid, "validation")],
                    early_stopping_rounds=50,
                )

                y_pred = booster.predict(valid)
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("rmse", rmse)

                mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

            elif model_type == "linear_regression":
                # Linear Regression training
                lr = LinearRegression()
                lr.fit(X_train, y_train)

                # Log model parameters
                mlflow.log_param("intercept", lr.intercept_)

                y_pred = lr.predict(X_val)
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("rmse", rmse)

                model_path = (
                    "./models/linear_regression_"
                    f"{train_features['year']}_{train_features['month']:02d}.pkl"
                )
                # Save and log the model
                with open(model_path, "wb") as f:
                    pickle.dump(lr, f)

                # Now log the artifact (after the file exists)
                mlflow.log_artifact(model_path, artifact_path="models_mlflow")

                # Also log using sklearn.log_model
                mlflow.sklearn.log_model(lr, artifact_path="models_sklearn")
        return run.info.run_id

    ##################################
    # DAG flow definition
    ##################################
    dates = get_dates()

    # Read training and validation data
    train_data = read_dataframe.override(task_id="read_training_data")(
        dates["train_year"],
        dates["train_month"],
    )
    val_data = read_dataframe.override(task_id="read_validation_data")(
        dates["val_year"],
        dates["val_month"],
    )

    # Create features
    train_features = dict_vectorize.override(task_id="convert_training_data")(train_data)
    val_features = dict_vectorize.override(task_id="convert_validation_data")(
        val_data, train_features["dict_vectorizer_path"]
    )

    # Extract targets separately
    train_target_path = extract_targets.override(task_id="extract_training_target")(train_data)
    val_target_path = extract_targets.override(task_id="extract_validation_target")(val_data)

    # Train model
    train_model(train_features, val_features, train_target_path, val_target_path)


# Create the DAG instance
dag_instance = train_regressor_nyc_taxi()
