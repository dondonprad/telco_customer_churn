import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function is responsible for data transformation
        '''

        cat_col = ['Charge  Amount','Tariff Plan', 'Age Group']
        num_col = ['Call  Failure', 'Subscription  Length', 'Seconds of Use','Frequency of use', 
                   'Frequency of SMS','Distinct Called Numbers','Age','Customer Value']
        
        logging.info("Categorical columns: %s", cat_col)
        logging.info("Numerical columns: %s", num_col)

        # Define the preprocessing steps for numerical and categorical features
        num_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
        cat_pipeline = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        logging.info("Numerical pipeline and categorical pipeline created successfully.")

        # Combine the numerical and categorical pipelines into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, num_col),
                ('cat', cat_pipeline, cat_col)
            ]
        )
        logging.info("Preprocessor object created successfully.")
        return preprocessor

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        This function is responsible for data transformation.
        It applies the preprocessing steps to the training and testing datasets.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Training and testing datasets loaded successfully.")

            X_train = train_df.drop(columns=['Churn'])
            y_train = train_df['Churn']
            X_test = test_df.drop(columns=['Churn'])
            y_test = test_df['Churn']

            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformation_object()

            # Fit the preprocessor on the training data and transform both train and test data
            X_train = preprocessor_obj.fit_transform(X_train)
            X_test = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("Data transformation completed successfully.")

            # Save the preprocessor object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_obj)

            return (train_arr, 
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')
    obj = DataTransformation()
    train_arr,test_arr,_ = obj.initiate_data_transformation(train_path,test_path)
    print(train_arr.shape, test_arr.shape)
    print("Data transformation completed successfully.")

    
