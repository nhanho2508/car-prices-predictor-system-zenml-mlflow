import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract interface for any model building strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Train a model using the provided training data.

        Parameters:
        - X_train: Feature matrix
        - y_train: Target vector

        Returns:
        - Trained scikit-learn model or pipeline
        """
        pass


# Concrete strategy: Linear Regression with standard scaling
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("Expected X_train to be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("Expected y_train to be a pandas Series.")

        logging.info("Setting up Linear Regression pipeline with standard scaling.")

        model_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])

        logging.info("Training Linear Regression model...")
        model_pipeline.fit(X_train, y_train)
        logging.info("Training completed.")

        return model_pipeline


# Context class: uses a model strategy to build and train models
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Model building strategy updated.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        logging.info("Starting model training using selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


if __name__ == "__main__":
    pass
