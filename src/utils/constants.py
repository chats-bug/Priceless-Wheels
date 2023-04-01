from enum import Enum


INDEX = 'usedCarSkuId'
TARGET = 'listed_price'
PROCESSED_DIR_PATH = '../../data/processed/'
CLEAN_DIR_PATH = '../../data/clean/'
RAW_DIR_PATH = '../../data/raw/'


class ImputationStrategy(Enum):
	MEAN = 'mean'
	MEDIAN = 'median'
	MODE = 'mode'
	CONSTANT = 'constant'
	KNN = 'knn'
	MICE = 'mice'