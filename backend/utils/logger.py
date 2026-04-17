import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name="factory_system"):
    # Ensure logs directory exists
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'app.log')
    
    logger = logging.getLogger(name)
    
    # Check if handlers are already configured to avoid duplicate logs in same process
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 5MB per file, keep max 5 backup files (25MB total)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=5
        )
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Also print to console
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent propagation to the root logger
        logger.propagate = False
        
    return logger
