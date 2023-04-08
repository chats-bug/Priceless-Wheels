import logging
import datetime
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.constants import *
from src.utils.data_loader import DataLoader


def prepare_data(
		filepath: str = None,
		train_size: float = 0.75,
		test_size: float = 0.15,
		validation_size: float = 0.1
) -> bool:
	"""
	Prepare the data for model training
	"""
	assert train_size + test_size + validation_size == 1, 'The sum of train, test and validation sizes should be 1'
	assert train_size > 0, 'Train size should be greater than 0'
	assert test_size > 0, 'Test size should be greater than 0'
	assert validation_size > 0, 'Validation size should be greater than 0'
	
	if filepath is None:
		dl = DataLoader(dir_path=PROCESSED_DIR_PATH)
		filepath = dl.get_latest_file(begins_with=PROCESSED_FILE_BEGIN)
		
	if filepath is None:
		raise FileNotFoundError('No processed file found')
		
	filepath = os.path.join(PROCESSED_DIR_PATH, filepath)
	success = False
	
	try:
		print(f"Reading processed file from : {filepath}")
		print('Preparing data for model training and testing...')
		df = pd.read_csv(filepath, index_col=INDEX)
		print(df.info())
		
		# Split the data into train, test and validation
		model_data, validation = train_test_split(df, test_size=validation_size, random_state=42)
		train, test = train_test_split(model_data, test_size=test_size, random_state=42)
		
		# Save the data in the train, test and validation directories
		train.to_csv(f"{TRAIN_DIR_PATH}{TRAIN_FILE_BEGIN}_{datetime.datetime.now().strftime(SAVE_DATE_TIME_FORMAT)}.csv")
		test.to_csv(f"{TEST_DIR_PATH}{TEST_FILE_BEGIN}_{datetime.datetime.now().strftime(SAVE_DATE_TIME_FORMAT)}.csv")
		validation.to_csv(f"{VALIDATION_DIR_PATH}{VALIDATION_FILE_BEGIN}_{datetime.datetime.now().strftime(SAVE_DATE_TIME_FORMAT)}.csv")
		
		print(f"Train data shape: {train.shape}")
		print(f"Test data shape: {test.shape}")
		print(f"Validation data shape: {validation.shape}")
		
		success = True
	except Exception as e:
		print(f"Error reading file: {filepath}")
		print(f"Please check the logs for more details")
		logging.basicConfig(
			filename='../data/logs/data_errors.log',
			level=logging.DEBUG,
			filemode='a',
			format='%(asctime)s %(message)s'
		)
		logging.exception(e)
	
	if success:
		print('Data preparation successfully completed')
	else:
		print('Data preparation failed')
		
	return success


def main():
	print('Preparing data for model training and testing...')
	prepare_data()
	

if __name__ == '__main__':
	main()
	