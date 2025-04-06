import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from .src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy
from zenml import step


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[dict, float]:
    """
    Evaluates a trained regression model using a defined evaluation strategy.

    Parameters:
    - trained_model: Pipeline including preprocessing and model
    - X_test: Features for evaluation
    - y_test: Ground truth target values

    Returns:
    - dict: Evaluation metrics (e.g., MSE, R2)
    - float: Mean Squared Error as primary metric
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("Expected X_test to be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("Expected y_test to be a pandas Series.")

    logging.info("Running preprocessing on test data...")
    X_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    logging.info("Evaluating model performance...")
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())
    metrics = evaluator.evaluate(
        trained_model.named_steps["model"],
        X_processed,
        y_test
    )

    if not isinstance(metrics, dict):
        raise ValueError("Expected evaluation metrics to be a dictionary.")

    mse = metrics.get("Mean Squared Error")
    logging.info(f"Evaluation completed. MSE: {mse:.2f}, Full metrics: {metrics}")

    return metrics, mse
