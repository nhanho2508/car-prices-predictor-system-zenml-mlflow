import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# Active experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker

# Define model metadata
model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a Linear Regression model wrapped in a preprocessing pipeline.

    Returns:
        Trained scikit-learn pipeline.
    """
    # Input validation
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Column selection
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {cat_cols.tolist()}")
    logging.info(f"Numerical columns: {num_cols.tolist()}")

    # Preprocessing pipelines
    num_transformer = SimpleImputer(strategy="mean")
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    # MLflow autologging
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Training Linear Regression pipeline...")
        pipeline.fit(X_train, y_train)
        logging.info("Model training complete.")

        # Log expected column names
        onehot = pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        onehot.fit(X_train[cat_cols])
        expected_cols = num_cols.tolist() + list(onehot.get_feature_names_out(cat_cols))
        logging.info(f"Pipeline expects columns: {expected_cols}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline
