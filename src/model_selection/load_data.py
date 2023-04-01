import os

import pandas as pd
from src.utils.constants import INDEX
from src.utils.utils import Utility as cutil
from src.utils.data_loader import DataLoader
from src.feature_engineering.feature_transformations import FeatureEngineeringTransformations


def transform_data(
		df: pd.DataFrame,
		copy: bool = True
	) -> pd.DataFrame:
	"""
	Apply all the transformations to the dataframe
	"""
	if copy:
		df = df.copy()
	
	fe_transformer = FeatureEngineeringTransformations(df)
	fe_transformer = fe_transformer.fit()

	df = fe_transformer.transform(df)


def get_transformed_data() -> pd.DataFrame:
	"""
	Load the latest processed data and apply all the transformations to it
	"""
	dl = DataLoader(dir_path='../../data/processed')
	filepath = dl.get_latest_file(begins_with='transformed_')
	df = pd.read_csv(filepath, index_col=INDEX)
	df = transform_data(df)
	return df


def main():
	print('Running transformations...')
	df = get_transformed_data()
	print(df.info())


if __name__ == "__main__":
	main()