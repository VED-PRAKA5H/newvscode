import sys  # Import the sys module for system-specific parameters and functions
from dataclasses import dataclass  # Import dataclass decorator for creating data classes
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation and analysis
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for applying different transformations to different columns
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.pipeline import Pipeline  # Import Pipeline for chaining transformations
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Import preprocessing techniques
from src.exception import CustomException  # Import custom exception for error handling
from src.logger import logging  # Import logging for tracking events
import os  # Import os for operating system dependent functionality
from src.utils import load_object  # Import utility function to load objects


@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation settings."""
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Path to save the preprocessor object


class DataTransformation:
    """Class for transforming data for machine learning models."""
    
    def __init__(self):
        """Initialize the DataTransformation class and its configuration."""
        self.data_transformation_config = DataTransformationConfig()  # Create an instance of the configuration class

    def get_data_transformer_object(self):
        """Create and return a data transformer object for preprocessing."""
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Create a pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with the median
                    ("scaler", StandardScaler())  # Scale numerical features
                ]
            )

            # Create a pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with the most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # Apply one-hot encoding to categorical features
                    ("scaler", StandardScaler(with_mean=False))  # Scale categorical features without centering
                ]
            )

            # Log the identified columns for debugging
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the numerical and categorical pipelines into a single preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply numerical pipeline
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Apply categorical pipeline
                ]
            )

            return preprocessor  # Return the preprocessor object

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Load data, apply transformations, and return the transformed data."""
        try:
            # Load training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"  # Define the target column name
            numerical_columns = ["writing_score", "reading_score"]  # Define numerical columns

            # Separate features and target variable for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Fit the preprocessor on the training data and transform it
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # Transform the testing data using the fitted preprocessor
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed features with the target variable for training and testing
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the preprocessor object to a file
            load_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                obj=preprocessing_obj
            )
            # Return the transformed training and testing data along with the file path of the preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)



