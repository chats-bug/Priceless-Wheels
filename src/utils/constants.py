from enum import Enum

INDEX = 'usedCarSkuId'

class ImputationStragety(Enum):
	MEAN = 'mean'
	MEDIAN = 'median'
	MODE = 'mode'
	CONSTANT = 'constant'
	KNN = 'knn'
	MICE = 'mice'