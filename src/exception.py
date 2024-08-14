import sys  # Import sys module for system-specific parameters and functions
import logging  # Import logging module for tracking and logging errors

def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information.

    Args:
        error (Exception): The exception object containing the error details.
        error_detail (sys): The sys module, used to extract traceback information.

    Returns:
        str: A formatted string containing detailed error information.
    """
    # Extract the traceback object from the error details
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Create a formatted error message with the filename, line number, and error message
    error_message = (
        "Error occurred in Python script name [{0}] "
        "line number [{1}] error_message [{2}]"
    ).format(file_name, exc_tb.tb_lineno, str(error))
    
    return error_message  # Return the formatted error message

class CustomException(Exception):
    """Custom exception class to handle errors."""
    
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with a detailed error message.

        Args:
            error_message (str): The error message to be logged.
            error_detail (sys): The sys module, used to extract traceback information.
        """
        super().__init__(error_message)  # Call the base class constructor
        # Get the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):  # Override the string representation of the exception
        """Return the detailed error message."""
        return self.error_message  # Return the formatted error message


                                                                # for check lets consider following case
        
# if __name__ == "__main__":
#     try:
#         a =1/0
#     except Exception as e:
#         logging.info("Divide my Zero")
#         raise CustomException(e, sys)

