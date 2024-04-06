import logging
import logging.config
import yaml

from src.utils.consts import LOG_CONF_FILE


def setup_logging():
    with open(LOG_CONF_FILE, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def get_default_logger() -> logging.Logger:
    # Setup logging using the configuration file
    setup_logging()

    # Create or get a logger
    return logging.getLogger(__name__)
