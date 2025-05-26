import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """
    file_path: str = os.path.join('artifacts', 'df_feature_drop_generic.csv')

class DataIngestion():
    """
    Class responsible for data ingestion.
    It loads data from a CSV file into a pandas DataFrame.
    """
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> pd.DataFrame:
        """
        Initiates the data ingestion process by loading data from a CSV file.

        Returns:
        - pd.DataFrame: The loaded DataFrame.
        """
        try:
            df = pd.read_csv(self.data_ingestion_config.file_path)
            return df

        except Exception as e:
            raise CustomException(sys, e)

  
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df = data_ingestion.initiate_data_ingestion()
    print("Data Ingestion completed successfully.")
    print(df.head())
    print("Data loaded successfully.")
