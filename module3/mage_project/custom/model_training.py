if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import uuid

@custom
def transform_custom(df, *args, **kwargs):
    """
    df: DataFrame returned by prepare_yellow (with 'duration' column)
    """
    run_uuid = uuid.uuid4()
    model_name = f"yellow_taxi_duration_{run_uuid}"
    mlflow.set_experiment(model_name)
    with mlflow.start_run():
        # 1) Define features
        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']

        # 2) Vectorize
        train_dicts = df[categorical + numerical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)

        # 3) Prepare target
        y_train = df.duration.values

        # 4) Fit Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # 5) Print the intercept
        print(f'Model intercept_: {lr.intercept_}')

        # 6) Log artifacts
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="model",
            registered_model_name="yellow_taxi_duration_model"
        )

    # 6) Return both the vectorizer and the model
    return {
        'dv': dv,
        'model': lr,
    }