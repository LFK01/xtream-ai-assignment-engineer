import os.path
from logging import config, Logger, getLogger, DEBUG, Formatter
from logging.handlers import RotatingFileHandler
import yaml

from src.utils.consts import LOG_CONF_FILE, LOG_DIR


def create_default_logger(e: Exception) -> Logger:
    # Create a logger
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)

    # Define the format for log messages
    formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create a file handler
    file_handler = RotatingFileHandler(filename=str(os.path.join(LOG_DIR, 'log_messages.log')),
                                       maxBytes=10 * 1024 * 1024,
                                       backupCount=5)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    logger.error(f'Using default file handler logger due to exception {e}')

    return logger


def get_logger(filepath: str = LOG_CONF_FILE) -> Logger:
    try:
        with open(filepath, 'r') as f:
            yaml_config = yaml.safe_load(f.read())
            config.dictConfig(yaml_config)

            # Create or get a logger
            return getLogger(__name__)
    except FileNotFoundError as e:
        logger = create_default_logger(e)
        return logger

