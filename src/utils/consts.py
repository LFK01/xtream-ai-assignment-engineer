import os

# DO NOT MOVE THIS FILE. It is used as a reference point to get the path to the root folder
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(MODULE_DIR, '..')
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
DIAMONDS_DATASETS_DIR = os.path.join(DATASETS_DIR, 'diamonds')
DIAMONDS_CSV_FILE = os.path.join(DIAMONDS_DATASETS_DIR, 'diamonds.csv')

CONF_DIR = os.path.join(ROOT_DIR, 'conf')
TRAIN_CONFIGURATION_FILE = os.path.join(CONF_DIR, 'train_conf.json')
INFER_CONFIGURATION_FILE = os.path.join(CONF_DIR, 'infer_conf.json')
LOG_CONF_FILE = os.path.join(CONF_DIR, 'log_conf.yaml')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'pytorch_models')

MODEL_FILENAME = 'model.pth'
PCA_PICKLE_FILENAME = 'pca.pickle'
MEAN_STD_DEV_FILENAME = 'mean_std.json'

TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'runs')
