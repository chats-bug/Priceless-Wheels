# Run this file to do the following:
# 1. Load the raw data
# 2. Clean the data and save the file as csv (code in src/data/cleaning.py)
# 3. Preprocess the data and save the file as csv (code in src/data/preprocessing.py)

from src.data.cleaning import run_cleaning_process
from src.data.preprocessing import run_transformations


def run_preprocessing_steps() -> str:
	print('Running preprocessing steps...')
	print('Step 1: Cleaning the data...')
	clean_file_path = run_cleaning_process()
	print(f"Cleaned data saved to {clean_file_path}")
	print('Step 2: Transforming the data...')
	processed_file_path = run_transformations()
	print(f"Transformed data saved to {processed_file_path}")
	
	return processed_file_path


def main():
	run_preprocessing_steps()


if __name__ == "__main__":
	main()
