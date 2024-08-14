import os  # Import os module for operating system dependent functionality
import sys  # Import sys module for system-specific parameters and functions
from src.exception import CustomException  # Import custom exception for error handling
from src.logger import logging  # Import logging for tracking events
import pandas as pd  # Import pandas for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data into training and testing sets
from dataclasses import dataclass  # Import dataclass decorator for creating data classes
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion settings."""
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save training data
    test_data_path: str = os.path.join('artifacts', "test.csv")    # Path to save testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")     # Path to save raw data

class DataIngestion:
    """Class for ingesting data from source to storage."""
    
    def __init__(self):
        """Initialize the DataIngestion class and its configuration."""
        self.ingestion_config = DataIngestionConfig()  # Create an instance of the configuration class

    def initiate_data_ingestion(self):
        """Method to initiate the data ingestion process."""
        logging.info("Entered the data ingestion method or component")  # Log the entry into the method
        try:
            # Load the dataset from the specified path into a DataFrame
            df = pd.read_csv('../../notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')  # Log successful reading of the dataset

            # Create the directory for raw data if it doesn't exist and save the raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # Save raw data as CSV

            logging.info("Train-test split initiated")  # Log the initiation of train-test split

            # Split the DataFrame into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Create the directory for training data if it doesn't exist and save the training set
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  # Save training data as CSV

            # Create the directory for testing data if it doesn't exist and save the testing set
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # Save testing data as CSV

            logging.info("Ingestion of the data is completed")  # Log completion of data ingestion
            return (
                self.ingestion_config.train_data_path,  # Return the path to the training data
                self.ingestion_config.test_data_path     # Return the path to the testing data
            )
        except Exception as e:
            # Raise a custom exception if an error occurs during data ingestion
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    obj = DataIngestion()  # Create an instance of the DataIngestion class
    train_data, test_data =obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data) # Call the method to initiate data ingestion
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

