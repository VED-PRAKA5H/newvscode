import sys
import logging  # use for tracking purpose
import os
from datetime import datetime

# Create a log file name with the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the logs directory
logs_directory = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_directory, exist_ok=True)

# Create the full log file path
LOG_FILE_PATH = os.path.join(logs_directory, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")

