import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


# Base class for data ingestion strategies
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest data from the specified file path.

        Parameters:
        - file_path (str): Path to the input file.

        Returns:
        - pd.DataFrame: Loaded data.
        """
        pass


# Strategy: Ingest from ZIP file containing one CSV
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("Expected a .zip file.")

        extract_dir = "tmp/extracted_data"
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        csv_files = [f for f in os.listdir(extract_dir) if f.endswith(".csv")]

        if not csv_files:
            raise FileNotFoundError("No CSV file found in the ZIP archive.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Specify the target file.")

        csv_path = os.path.join(extract_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        return df


# Factory to return the appropriate DataIngestor
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Get the right DataIngestor based on file type.

        Parameters:
        - file_extension (str): Extension like '.zip', '.csv', etc.

        Returns:
        - DataIngestor instance.
        """
        if file_extension == ".zip":
            return ZipDataIngestor()
        raise ValueError(f"No ingestor implemented for '{file_extension}' files.")



if __name__ == "__main__":
    pass
