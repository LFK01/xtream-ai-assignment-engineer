import os
from src.utils.logger import get_default_logger
from src.utils.consts import LOG_DIR, SAVED_MODELS_DIR, TENSORBOARD_DIR, DIAMONDS_CSV_FILE
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

    logger = get_default_logger()

    df = DataProcessor.load_data(DIAMONDS_CSV_FILE)

    model = NeuralNetworkPredictor(headers_dict=COLS_DICT,
                                   hidden_size=2 ** 10,
                                   logger=logger)

    indices_to_select = [2797, 2986, 4132, 4287, 4259]
    result = model.infer(df.iloc[indices_to_select])
    logger.info('\n'.join([f'Pred: {pred} Target: {target} Error: {abs(pred - target)}'
                           for pred, target in zip(result, df.iloc[indices_to_select]['price'].values)]))
