import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

file_path = os.path.join('artifacts','df_feature_drop_generic.csv')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df

    except Exception as e:
        raise CustomException(sys,e)

""""   
if __name__ == "__main__":
    df = load_data(file_path)
    print(df.head())
    print("Data loaded successfully.")
"""