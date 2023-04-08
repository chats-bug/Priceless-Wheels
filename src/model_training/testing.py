import os

import joblib
import numpy as np

from src.model_selection.load_data import load_validation_data
from src.utils.constants import TARGET, INDEX, MODEL_DIR_PATH
from src.utils.data_loader import DataLoader


def main():
	# Load the validation data
	df = load_validation_data(verbose=False)
	X = df.drop(columns=TARGET)
	y = df[TARGET]
	
	# Load the file names of the models
	dl = DataLoader(MODEL_DIR_PATH)
	catboost_pipeline_filename = dl.get_latest_file(begins_with='catboost_pipeline')
	lightgbm_pipeline_filename = dl.get_latest_file(begins_with='lightgbm_pipeline')
	if catboost_pipeline_filename is None or lightgbm_pipeline_filename is None:
		raise FileNotFoundError('No model file found. Train the models first')
	
	# Get the full path of the models
	catboost_pipeline_filepath = os.path.join(MODEL_DIR_PATH, catboost_pipeline_filename)
	lightgbm_pipeline_filepath = os.path.join(MODEL_DIR_PATH, lightgbm_pipeline_filename)
	
	# Load the models
	catboost_pipeline = joblib.load(catboost_pipeline_filepath)
	lightgbm_pipeline = joblib.load(lightgbm_pipeline_filepath)
	
	# Predict the target
	catboost_pred = np.exp(catboost_pipeline.predict(X))
	lightgbm_pred = np.exp(lightgbm_pipeline.predict(X))
	
	# Average the predictions
	avg_pred = (catboost_pred + lightgbm_pred) / 2
	
	# Print the scores: MAE and MAPE
	mae = (abs(y - avg_pred)).mean()
	mape = (abs(y - avg_pred) / y).mean()
	
	print(f"MAE: {mae}")
	print(f"MAPE: {mape}")
	

if __name__ == '__main__':
	main()