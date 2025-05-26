import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split

@dataclass
class DataPreprocesingConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')

class DataPreprocesing():
    def __init__(self):
        self.data_preprocession_config = DataPreprocesingConfig()

    def initiate_data_preprocessing(self, df: pd.DataFrame) -> None:
        """
        This function is responsible for data preprocessing.
        It splits the data into training and testing sets and saves them to CSV files.
        """
        try:
            logging.info("Starting data preprocessing...")

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into training and testing sets.")

            # Save the training and testing sets to CSV files
            train_set.to_csv(self.data_preprocession_config.train_data_path, index=False)
            test_set.to_csv(self.data_preprocession_config.test_data_path, index=False)
            logging.info("Training and testing sets saved successfully.")
            return self.data_preprocession_config.train_data_path, self.data_preprocession_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_preprocessing = DataPreprocesing()
    df = pd.read_csv(os.path.join('artifacts', 'df_feature_drop_generic.csv'))
    train_path, test_path = data_preprocessing.initiate_data_preprocessing(df)
    print("Data Preprocessing completed successfully.")
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")