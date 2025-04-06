import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Base interface for model evaluation strategies
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate a trained model and return metrics.

        Parameters:
        - model: Trained regressor
        - X_test: Test feature set
        - y_test: Ground truth targets

        Returns:
        - Dictionary of evaluation metrics
        """
        pass


# Concrete strategy for regression evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info("Generating predictions...")
        y_pred = model.predict(X_test)

        logging.info("Calculating regression metrics...")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            "Mean Squared Error": mse,
            "R-Squared": r2
        }

        logging.info(f"Evaluation results: {results}")
        return results


# Context class to evaluate models using a selected strategy
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initialize with a model evaluation strategy.

        Parameters:
        - strategy: Strategy object implementing evaluation logic
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Change the current evaluation strategy.

        Parameters:
        - strategy: New evaluation strategy
        """
        logging.info("Updated model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the model using the current strategy.

        Parameters:
        - model: Trained model
        - X_test: Feature set
        - y_test: Ground truth labels

        Returns:
        - Dictionary of evaluation metrics
        """
        logging.info("Starting model evaluation...")
        return self._strategy.evaluate(model, X_test, y_test)



if __name__ == "__main__":
    pass
