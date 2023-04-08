import os

import pandas as pd
from src.utils.constants import *
from src.utils.data_loader import DataLoader


def load_training_data(
		filepath: str = None,
		verbose: bool = True,
) -> pd.DataFrame | None:
	"""
	Load the training data
	
	Returns
	-------
		pd.DataFrame | None
			The training data or None if the file is not found
	"""
	if filepath is None:
		dl = DataLoader(TRAIN_DIR_PATH)
		filename = dl.get_latest_file(begins_with=TRAIN_FILE_BEGIN)
		if filename is None:
			raise FileNotFoundError('No training file found. Prepare the data first')
		filepath = os.path.join(TRAIN_DIR_PATH, filename)
	
	print(f"Reading training file from : {filepath}")
	try:
		df = pd.read_csv(filepath, index_col=INDEX)
		if verbose:
			print(df.info())
		return df
	except Exception as e:
		print(f"Error reading file: {filepath}")
		print(e)
		return None


def load_testing_data(
		filepath: str = None,
		verbose: bool = True,
) -> pd.DataFrame | None:
	"""
	Load the testing data
	
	Returns
	-------
		pd.DataFrame
			The testing data or None if the file is not found
	"""
	if filepath is None:
		dl = DataLoader(TEST_DIR_PATH)
		filename = dl.get_latest_file(begins_with=TEST_FILE_BEGIN)
		if filename is None:
			raise FileNotFoundError('No testing file found. Prepare the data first')
		filepath = os.path.join(TEST_DIR_PATH, filename)
	
	print(f"Reading testing file from : {filepath}")
	try:
		df = pd.read_csv(filepath, index_col=INDEX)
		if verbose:
			print(df.info())
		return df
	except Exception as e:
		print(f"Error reading file: {filepath}")
		print(e)
		return None


def load_validation_data(
		filepath: str = None,
		verbose: bool = True,
) -> pd.DataFrame | None:
	"""
	Load the validation data
	
	Returns
	-------
		pd.DataFrame
			The validation data or None if the file is not found
	"""
	if filepath is None:
		dl = DataLoader(VALIDATION_DIR_PATH)
		filename = dl.get_latest_file(begins_with=VALIDATION_FILE_BEGIN)
		if filename is None:
			raise FileNotFoundError('No validation file found. Prepare the data first')
		filepath = os.path.join(VALIDATION_DIR_PATH, filename)
	
	print(f"Reading validation file from : {filepath}")
	try:
		df = pd.read_csv(filepath, index_col=INDEX)
		if verbose:
			print(df.info())
		return df
	except Exception as e:
		print(f"Error reading file: {filepath}")
		print(e)
		return None


def load_train_test_valid_data(
		train_path: str = None,
		test_path: str = None,
		validation_path: str = None,
		verbose: bool = False,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
	"""
	Load the training, testing and validation data
	
	Returns
	-------
		(pd.DataFrame, pd.DataFrame, pd.DataFrame) or None
			The training, testing and validation data respectively or None if any of the files is not found
	"""
	
	train = load_training_data(filepath=train_path, verbose=verbose)
	test = load_testing_data(filepath=test_path, verbose=verbose)
	validation = load_validation_data(filepath=validation_path, verbose=verbose)
	
	if train is None or test is None or validation is None:
		return None
	
	return train, test, validation
