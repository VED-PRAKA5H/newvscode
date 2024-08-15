# utils.py
# This module contains common utility functions used throughout the project

import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score  # Import r2_score for model evaluation
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


# Example object to serialize
data = {
    'name': 'Alice',
    'age': 30,
    'skills': ['Python', 'Data Science', 'Machine Learning']
}

def load_object(file_path, obj):
    """
    Saves an object to a specified file path using the dill library.

    Args:
        file_path (str): The file path where the object will be saved.
        obj (object): The object to be saved.

    Raises:
        CustomException: If an exception occurs during the saving process.
    """
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Open the file in binary write mode
        with open(file_path, "wb") as file_obj:
            # Use dill to dump the object to the file
            dill.dump(obj, file_obj)
    except Exception as e:
        # Raise a custom exception with the original exception and system information
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates multiple machine learning models and returns their performance scores.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        X_test (np.ndarray): Testing feature data.
        y_test (np.ndarray): Testing target data.
        models (dict): A dictionary of models to evaluate.

    Returns:
        dict: A dictionary containing model names and their corresponding test scores.

    Raises:
        CustomException: If an exception occurs during model evaluation.
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train, y_train)
            
            # model.fit(X_train, y_train)  # Train the model
            
            # Make predictions
            y_train_predictions = model.predict(X_train)  # Predict on training data
            y_test_predictions = model.predict(X_test)  # Predict on testing data
            
            # Calculate R² scores
            train_model_score = r2_score(y_train, y_train_predictions)
            test_model_score = r2_score(y_test, y_test_predictions)
            
            # Store the test score in the report
            report[list(models.keys())[i]] = test_model_score
        
        return report  # Return the report with model scores
    except Exception as e:
        raise CustomException(e, sys)  # Raise a custom exception if an error occurs


