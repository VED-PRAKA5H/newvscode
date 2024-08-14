import sys
import logging  # Import logging module for tracking application events
import os
from datetime import datetime  # Import datetime to work with timestamps

# Create a log file name using the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the logs directory path by joining the current working directory with "logs"
logs_directory = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't already exist
os.makedirs(logs_directory, exist_ok=True)

# Create the full path for the log file by joining the logs directory with the log file name
LOG_FILE_PATH = os.path.join(logs_directory, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Specify the log file path
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define the log message format
    level=logging.INFO,  # Set the logging level to INFO
)

# Entry point of the script
if __name__ == "__main__":
    # Log a message indicating that logging has started
    logging.info("Logging has started")