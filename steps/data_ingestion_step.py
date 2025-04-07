import pandas as pd
from .src.ingest_data import DataIngestorFactory
from zenml import step


@step()
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    Step to ingest data from a .zip file using the corresponding DataIngestor.

    Parameters:
    - file_path (str): Path to the zip file

    Returns:
    - pd.DataFrame: Loaded data
    """
    # Define the file type (hardcoded as ZIP for this pipeline)
    ext = ".zip"

    # Select and initialize the appropriate ingestor
    ingestor = DataIngestorFactory.get_data_ingestor(ext)

    # Load and return the data
    df = ingestor.ingest(file_path)
    return df
