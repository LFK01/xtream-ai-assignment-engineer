import os
import json
from src.utils.logger_utils import get_logger, create_default_logger
from src.utils.consts import LOG_DIR, SAVED_MODELS_DIR, TENSORBOARD_DIR, TRAIN_CONFIGURATION_FILE, ROOT_DIR
from src.data_utils.preprocess import DataProcessor, COLS_DICT
from src.model.model import NeuralNetworkPredictor

if __name__ == '__main__':
    # Load the JSON file
    with open(TRAIN_CONFIGURATION_FILE, 'r') as file:
        config_dict = json.load(file)

    try:
        logger = get_logger(filepath=config_dict['logger']['conf_file'])
    except KeyError as e:
        logger = create_default_logger(e)

    # Check debugging and file storage directories exist otherwise create them
    if not os.path.exists(LOG_DIR):
        logger.debug(f'Making dir: {LOG_DIR}')
        os.mkdir(LOG_DIR)
    if not os.path.exists(SAVED_MODELS_DIR):
        logger.debug(f'Making dir: {SAVED_MODELS_DIR}')
        os.mkdir(SAVED_MODELS_DIR)
    if not os.path.exists(TENSORBOARD_DIR):
        logger.debug(f'Making dir: {TENSORBOARD_DIR}')
        os.mkdir(TENSORBOARD_DIR)

    df = DataProcessor.load_data(filepath=config_dict['dataset']['csv_file'])

    logger.debug(f'Loaded data from the file: {config_dict['dataset']['csv_file']}')

    model = NeuralNetworkPredictor(headers_dict=COLS_DICT,
                                   data=df,
                                   hidden_size=2 ** 10,
                                   logger=logger)

    model.prepare_datasets()
    model.train()
    model.test()
