import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Base strategy for outlier detection
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify outliers in the given DataFrame.

        Returns:
            pd.DataFrame: Boolean DataFrame where True indicates an outlier.
        """
        pass


# Strategy: Z-score based outlier detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Detecting outliers using Z-score (threshold={self.threshold})")
        z_scores = np.abs((df - df.mean()) / df.std())
        return z_scores > self.threshold


# Strategy: IQR-based outlier detection
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))


# Context: Outlier detector using a chosen strategy
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Outlier detection strategy updated.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running outlier detection.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method: str = "remove") -> pd.DataFrame:
        outliers = self.detect_outliers(df)

        if method == "remove":
            logging.info("Removing rows with any detected outliers.")
            return df[~outliers.any(axis=1)]

        elif method == "cap":
            logging.info("Capping outliers at 1st and 99th percentiles.")
            lower = df.quantile(0.01)
            upper = df.quantile(0.99)
            return df.clip(lower=lower, upper=upper, axis=1)

        logging.warning(f"Unknown handling method '{method}'. No changes applied.")
        return df

    def visualize_outliers(self, df: pd.DataFrame, features: list[str]):
        logging.info(f"Visualizing boxplots for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Boxplot visualization completed.")



if __name__ == "__main__":
    pass
