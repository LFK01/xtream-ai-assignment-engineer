import os
import json
from src.utils.logger_utils import get_logger
from src.utils.consts import LOG_DIR, SAVED_MODELS_DIR, TENSORBOARD_DIR, INFER_CONFIGURATION_FILE
from src.data_utils.preprocess import DataProcessor, COLS_DICT
from src.model.model import NeuralNetworkPredictor

if __name__ == '__main__':
    # Check debugging and file storage directories exist otherwise create them
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.exists(SAVED_MODELS_DIR):
        os.mkdir(SAVED_MODELS_DIR)
    if not os.path.exists(TENSORBOARD_DIR):
        os.mkdir(TENSORBOARD_DIR)

    # Load the JSON file
    with open(INFER_CONFIGURATION_FILE, 'r') as file:
        config_dict = json.load(file)

    logger = get_logger(filepath=config_dict['logger']['conf_file'])

    df = DataProcessor.load_data(filepath=config_dict['dataset']['csv_file'])

    model = NeuralNetworkPredictor(headers_dict=COLS_DICT,
                                   hidden_size=2 ** 10,
                                   logger=logger)

    result = model.infer(df)

    df[COLS_DICT['price']] = result
    logger.info(df.head())

    df.to_csv(config_dict['output']['file'])
