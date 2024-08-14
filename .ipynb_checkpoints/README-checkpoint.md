# Github And Code Set up
1. setup the github `repository`
   * new environment  (you can do this by installing conda)
   * setup.py
   * requirements.txt

## create envienvironment 
```conda create --name myenv python==3.9```

# 2. create setup.py file

## why to use setup.py file in machine learning project
* it is set of instructions
* it tells python things like what your project is called. who made it what it needs yo work
* without it people might have hard time to figuring out how to use
* it's like recipe for setting up your project on someone else's computer


3. # create src folder and build the package
    * in this create init.py file
    * ## why we use init.py
    * `Package Initialization`: When Python sees an init.py file inside a directory, it treats that directory as a package. It helps organize your code into logical units. 
    * `Namespace Package`:  If you have a large project split across multiple directories or locations, each having its own init.py, Python combines them all together when you import the package. This allows you to spread your code across different parts of your project without conflicts.m.


4. # create requirements.txt
    * ## `-e` stand for specifies that the package should be installed in editable mode.
    * ## and `.` for current directory
    * both are use in `pip` not in `conda` (for conda: Remove the -e . line)
```
pandas
numpy
matplotlib
seaborn
scikit-learn
-e .
``` 


5. ## now run `python setup.py install` in cmd of youyour virtual environment
   ## now you can use `pip freeze > requirements.txt` for conda `conda list --export > requirements.txt`


# project stucture, logging and exceptionn handling 
* ## create folder (components) in src folder then create following python file
  * __init__.py
  * data_ingestion.py `for data reading`
  * data_transformation.py `for EDA`
  * model_trainer.py `for Model training`

    
* ## now create folder (pipeline) in src folder then create following python file
  * __init__.py
  * train_pipeline.py
  * predict_pipeline.py



* ## create python files in src
  * utils.py
  * logger.py
  * exception.py

# Start project EDA and Model Training
## create  folder notebook inside this folder data.csv
## Data Ingestion
```
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('../../notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            pass
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

```

## Data Transformation Implementation Using Pipelines

```
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """This function is responsible for data transformation"""
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
```

# util.py
```
# utils is used for common functions
import os
import sys
import numpy as np  # Corrected alias for numpy
import pandas as pd
import dill  # Removed incorrect alias
from src.exception import CustomException

def save_object(file_path, obj):
    """Saves an object to a specified file path using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)  # Corrected the usage of os.makedirs
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Corrected indentation
    except Exception as e:
        raise CustomException(e, sys)  # Raising custom exception

```

# Model Training And Evauating Component
## model_trainer.py
```
import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression  # Corrected import statement
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor  # Corrected import statement
from xgboost import XGBRegressor  # Corrected import statement
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """Configuration class for model training settings."""
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:  # Corrected class name
    def __init__(self):
        """Initialize the ModelTrainer class and its configuration."""
        self.model_trainer_config = ModelTrainerConfig()  # Corrected initialization

    def initiate_model_trainer(self, train_array, test_array):  # Removed unnecessary space
        """Method to initiate the model training process."""
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            # Define models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # Evaluate models and get their performance metrics
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            # Get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]  # Added closing bracket
            best_model = models[best_model_name]  # Corrected assignment

            # Check if the best model score is below the threshold
            if best_model_score < 0.6:
                raise CustomException("No model met the performance criteria (R² < 0.6)")  # Improved error message
            
            logging.info("Best found model on both training and testing dataset")
            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )
            
            # Make predictions using the best model
            predicted = best_model.predict(X_test)
            # Calculate the R² score of the predictions
            r2_square = r2_score(y_test, predicted)
            
            return r2_square  # Return the R² score
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
```

