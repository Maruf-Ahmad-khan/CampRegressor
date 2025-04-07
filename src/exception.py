import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Captures detailed error information including script name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()  # Get traceback details
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get file name
    line_number = exc_tb.tb_lineno  # Get line number

    error_message = "Error occurred in Python script [{0}], at line [{1}]: {2}".format(
        file_name, line_number, str(error)
    )
    
    return error_message

class CustomException(Exception):
    """
    Custom exception class to log detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

# Example Usage
if __name__ == "__main__":
     
    try:
        # Example error: Division by zero
        result = 10 / 0  
    except Exception as e:
        logging.error(CustomException(e, sys))  # Log the detailed error message
        print(CustomException(e, sys))  # Print the error message
