from enum import Enum


INDEX = 'usedCarSkuId'
TARGET = 'listed_price'
PROCESSED_DIR_PATH = '../../data/processed/'
PROCESSED_FILE_BEGIN = 'transformed'
CLEAN_DIR_PATH = '../../data/clean/'
CLEAN_FILE_BEGIN = 'cleaned'
RAW_DIR_PATH = '../../data/raw/'
RAW_FILE_BEGIN = 'cardekho_cars'
TRAIN_DIR_PATH = '../../data/train/'
TRAIN_FILE_BEGIN = 'train'
TEST_DIR_PATH = '../../data/test/'
TEST_FILE_BEGIN = 'test'
VALIDATION_DIR_PATH = '../../data/validation/'
VALIDATION_FILE_BEGIN = 'validation'
MODEL_DIR_PATH = '../../data/models/'

SAVE_DATE_TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'
RAW_SAVE_DATE_TIME_FORMAT = '%Y_%m_%d_%H_%M_%S'  # The raw data is saved in this format


class ImputationStrategy(Enum):
	MEAN = 'mean'
	MEDIAN = 'median'
	MODE = 'mode'
	CONSTANT = 'constant'
	KNN = 'knn'
	MICE = 'mice'