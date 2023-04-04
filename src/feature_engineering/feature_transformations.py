from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


from src.utils.constants import INDEX, TARGET


class FeatureEngineeringTransformations(BaseEstimator, TransformerMixin):
	"""
	This class contains all the recommended feature engineering transformations for the dataset.
	
	Parameters
	----------
		df: pd.DataFrame
			The dataframe to be transformed
		object_cols: list
			The list of columns that contain the car features like top_features, comfort_features, etc. that need to be transformed
	"""
	
	def __init__(
			self,
			object_cols=None,
			target=TARGET
	):
		self.feature_prices = None
		self.df = None
		if object_cols is None:
			object_cols = [
				'top_features',
				'comfort_features',
				'interior_features',
				'exterior_features',
				'safety_features'
			]
		self.object_cols = object_cols
		self.target = target
	
	def _car_object_feature_dict(self) -> dict:
		unique_feature_scores = dict()
		for col in self.object_cols:
			for _, row in self.df.iterrows():
				feature_list = literal_eval(row[col])
				for feature in feature_list:
					if feature in unique_feature_scores.keys():
						unique_feature_scores[feature][1] += 1
						unique_feature_scores[feature][0] += row[self.target]
					else:
						unique_feature_scores[feature] = [row[self.target], 1]
		
		return unique_feature_scores
	
	def _map_object_cols_to_scores(self, x: str) -> float:
		feature_list = literal_eval(x)
		feature_score = 0
		for feature in feature_list:
			if feature in self.feature_prices.keys():
				feature_score += self.feature_prices[feature][0] / self.feature_prices[feature][1]
			else:
				feature_score += 0
		return feature_score
	
	def _car_object_feature_transformation(self, df) -> pd.DataFrame:
		if self.feature_prices is None:
			raise Exception('Please fit the transformer first')
		
		for col in self.object_cols:
			df[f'{col}_score'] = df[col].apply(self._map_object_cols_to_scores)
			df.drop(col, axis=1, inplace=True)
			# Replace zero scores with nan
			df[f'{col}_score'] = df[f'{col}_score'].replace(0, np.nan)
		return df
	
	def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEngineeringTransformations':
		self.df = X.copy()
		self.feature_prices = self._car_object_feature_dict()
		return self
	
	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		check_is_fitted(self, 'feature_prices')
		
		# Transform the object columns to scores
		X = X.copy()
		X = self._car_object_feature_transformation(X)
		return X