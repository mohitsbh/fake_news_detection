#!/usr/bin/env python
"""
Logging and error handling utilities for the fake news detection system.

This module provides functions for setting up logging and handling errors
in a consistent way across the application.
"""

import os
import sys
import logging
import traceback
from datetime import datetime

# Default log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')

def setup_logger(name, log_file=None, level=logging.INFO, console_output=True):
    """Set up a logger with file and/or console output.
    
    Args:
        name (str): Logger name.
        log_file (str, optional): Path to log file. If None, no file logging is set up.
        level (int, optional): Logging level. Default is logging.INFO.
        console_output (bool, optional): Whether to output logs to console. Default is True.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_file is provided
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_default_logger(name=None, console_output=True):
    """Get a default logger that logs to a file in the logs directory.
    
    Args:
        name (str, optional): Logger name. If None, uses the calling module's name.
        console_output (bool, optional): Whether to output logs to console. Default is True.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    if name is None:
        # Get the name of the calling module
        frame = sys._getframe(1)
        name = frame.f_globals['__name__']
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # Create log file name based on date and module name
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(LOG_DIR, f"{date_str}_{name}.log")
    
    return setup_logger(name, log_file, console_output=console_output)

class ErrorHandler:
    """Class for handling and logging errors."""
    
    def __init__(self, logger=None):
        """Initialize the error handler.
        
        Args:
            logger (logging.Logger, optional): Logger to use. If None, creates a default logger.
        """
        self.logger = logger or get_default_logger('error_handler')
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle an exception by logging it.
        
        Args:
            exc_type: Exception type.
            exc_value: Exception value.
            exc_traceback: Exception traceback.
        
        Returns:
            bool: True if the exception was handled, False otherwise.
        """
        # Format the exception
        exception_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Log the exception
        self.logger.error(f"Unhandled exception: {exception_str}")
        
        return True
    
    def install_global_handler(self):
        """Install this handler as the global exception handler."""
        sys.excepthook = self.handle_exception
        self.logger.info("Global exception handler installed")

def log_function_call(logger):
    """Decorator to log function calls.
    
    Args:
        logger (logging.Logger): Logger to use.
    
    Returns:
        function: Decorated function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned {result}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

def log_execution_time(logger):
    """Decorator to log function execution time.
    
    Args:
        logger (logging.Logger): Logger to use.
    
    Returns:
        function: Decorated function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds with error: {str(e)}")
                raise
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage
    logger = get_default_logger('example')
    logger.info("This is an example log message")
    
    # Install global exception handler
    error_handler = ErrorHandler(logger)
    error_handler.install_global_handler()
    
    # Example function with logging decorators
    @log_function_call(logger)
    @log_execution_time(logger)
    def example_function(x, y):
        logger.info(f"Processing {x} and {y}")
        return x + y
    
    result = example_function(3, 4)
    logger.info(f"Result: {result}")