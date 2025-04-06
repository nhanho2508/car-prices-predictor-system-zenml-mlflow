from abc import ABC, abstractmethod
import pandas as pd


# --- Abstract Strategy ---
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering steps and return the updated DataFrame."""
        pass


# --- Concrete Strategy ---
class CarFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def __init__(self, current_year: int = 2025):
        self.current_year = current_year

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Create 'age' column from 'year'
        df['age'] = self.current_year - df['year']

        # 2. Drop 'year' column
        df.drop(['year'], axis=1, inplace=True)

        # 3. Encode 'owner'
        df['owner'] = df['owner'].replace({
            'First Owner': 1,
            'Second Owner': 2,
            'Third Owner': 3
        })

        return df


# --- Context Class ---
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.apply(df)
