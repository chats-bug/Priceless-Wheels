import datetime
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils.validation import check_is_fitted

from src.model_selection.lightGBM_model import LightGBMModel
from src.model_selection.load_data import load_train_test_valid_data
from src.utils.constants import TARGET, MODEL_DIR_PATH, SAVE_DATE_TIME_FORMAT
from src.model_selection.catboost_model import CatBoostModel
from sklearn import set_config
set_config(transform_output="pandas")


class CatBoostRegModel:
	def __init__(self, transform_target: bool = False, transform: callable = np.log, inverse_transform: callable = np.exp):
		self.pipeline = None
		self.transform_target = transform_target
		self.transform = transform
		self.inverse_transform = inverse_transform
		
		self.model_class = CatBoostModel()
		self.model = self.model_class.get_model()
		self.preprocessor = None
		self.num_features_after_transformation = None
		self.cat_features_after_transformation = None
	
	def fit(self, X: pd.DataFrame, y: pd.Series):
		X = X.copy()
		y = y.copy()

		# Initialize the preprocessor
		self.preprocessor = self.model_class.get_preprocessor(X)
		# cat_cols = [f'cat__{col}' for col in X.select_dtypes(include=['object', 'category']).columns.tolist()]
		
		# Transform the target
		if self.transform_target:
			y = self.transform(y)
			
		# Make the pipeline
		self.pipeline = pipeline.Pipeline(
			steps=[
				('preprocessor', self.preprocessor),
				('model', self.model)
			]
		)
		
		# Fit the pipeline
		self.pipeline.fit(X, y, model__cat_features=self.model_class.cat_features_after_transformation)
		return self
	
	def predict(self, X: pd.DataFrame) -> np.ndarray:
		check_is_fitted(self, ['pipeline'])
		
		X = X.copy()
		# Predict
		y_preds = self.pipeline.predict(X)
		
		# Inverse transform the target
		if self.transform_target:
			y_preds = self.inverse_transform(y_preds)
		
		return y_preds
	
	def save_model(self, path: str):
		check_is_fitted(self, ['pipeline'])
		joblib.dump(self.pipeline, path)


class LightGBMRegModel:
	def __init__(self, transform_target: bool = False, transform: callable = np.log, inverse_transform: callable = np.exp):
		self.pipeline = None
		self.transform_target = transform_target
		self.transform = transform
		self.inverse_transform = inverse_transform
		
		self.model_class = LightGBMModel()
		self.model = self.model_class.get_model()
		self.preprocessor = None
		self.num_features_after_transformation = None
		self.cat_features_after_transformation = None
	
	def fit(self, X: pd.DataFrame, y: pd.Series):
		X = X.copy()
		y = y.copy()

		# Initialize the preprocessor
		self.preprocessor = self.model_class.get_preprocessor(X)
		
		# Transform the target
		if self.transform_target:
			y = self.transform(y)
			
		# Make the pipeline
		self.pipeline = pipeline.Pipeline(
			steps=[
				('preprocessor', self.preprocessor),
				('model', self.model),
			]
		)
		
		# Fit the pipeline
		self.pipeline.fit(X, y)
		return self
	
	def predict(self, X: pd.DataFrame) -> np.ndarray:
		check_is_fitted(self, ['pipeline'])
		
		X = X.copy()
		# Predict
		y_preds = self.pipeline.predict(X)
		
		# Inverse transform the target
		if self.transform_target:
			y_preds = self.inverse_transform(y_preds)
		
		return y_preds
	
	def save_model(self, path: str):
		check_is_fitted(self, ['pipeline'])
		joblib.dump(self.pipeline, path)


def main():
	# Load the data
	train, test, _ = load_train_test_valid_data()
	X_train, y_train = train.drop(columns=TARGET).reset_index(drop=True), train[TARGET].reset_index(drop=True)
	X_test, y_test = test.drop(columns=TARGET).reset_index(drop=True), test[TARGET].reset_index(drop=True)
	
	# Append the rows in X_test to X_train
	X_train = pd.concat([X_train, X_test], ignore_index=True)
	y_train = pd.concat([y_train, y_test], ignore_index=True)
	
	# Initialize the models
	catboost_model = CatBoostRegModel(transform_target=True)
	lightgbm_model = LightGBMRegModel(transform_target=True)
	
	# Fit the models
	print('Fitting the models...')
	print('CatBoost')
	catboost_model.fit(X_train, y_train)
	print('LightGBM')
	lightgbm_model.fit(X_train, y_train)
	
	# Predict
	print('Predicting...')
	catboost_preds = catboost_model.predict(X_test)
	lightgbm_preds = lightgbm_model.predict(X_test)
	combined_preds = (catboost_preds + lightgbm_preds) / 2
	# Evaluate
	catboost_scores = {
		'mae': mean_absolute_error(y_test, catboost_preds),
		'mape': mean_absolute_percentage_error(y_test, catboost_preds),
	}
	lightgbm_scores = {
		'mae': mean_absolute_error(y_test, lightgbm_preds),
		'mape': mean_absolute_percentage_error(y_test, lightgbm_preds),
	}
	combined_scores = {
		'mae': mean_absolute_error(y_test, combined_preds),
		'mape': mean_absolute_percentage_error(y_test, combined_preds),
	}
	
	print(f'CatBoost scores: {catboost_scores}')
	print(f'LightGBM scores: {lightgbm_scores}')
	print(f'Combined scores: {combined_scores}')
	
	res = input('Save the models? (y/n): ')
	if res.lower() != 'y':
		return
	
	# Save the models
	print('Saving the models...')
	file_ext = datetime.datetime.now().strftime(SAVE_DATE_TIME_FORMAT)
	catboost_model.save_model(path=os.path.join(MODEL_DIR_PATH, f'catboost_pipeline_{file_ext}.pkl'))
	lightgbm_model.save_model(path=os.path.join(MODEL_DIR_PATH, f'lightgbm_pipeline_{file_ext}.pkl'))
	print('Done!')


if __name__ == '__main__':
	main()
