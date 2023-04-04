from __future__ import annotations

import datetime
import os
import numpy as np
import pandas as pd

from src.utils.constants import INDEX, TARGET
from src.utils.data_loader import DataLoader
from src.utils.constants import CLEAN_DIR_PATH, CLEAN_FILE_BEGIN, PROCESSED_DIR_PATH, PROCESSED_FILE_BEGIN


class Transformations:
	"""
	This class contains all the recommended transformation functions for the dataset.
	For a deeper dive into the transformations and visualizations, please refer to the notebooks 'Data Exploration I, II' in the notebooks directory.
	"""
	
	def loc_transformation(df: pd.DataFrame):
		"""
		Drop the column 'loc' from the dataframe
		"""
		df.drop('loc', axis=1, inplace=True)
		return
	
	def myear_transformation(df: pd.DataFrame):
		"""
		Remove all rows where the year is less than 2005
		"""
		df.drop(df[df['myear'] < 2005].index, inplace=True)
		return df
	
	def images_transformation(df: pd.DataFrame):
		"""
		Drop the column 'images' from the dataframe
		"""
		df.drop('images', axis=1, inplace=True)
		return
	
	def imgCount_transformation(df: pd.DataFrame):
		"""
		Drop the column 'imgCount' from the dataframe
		"""
		df.drop('imgCount', axis=1, inplace=True)
		return
	
	def threesixty_transformation(df: pd.DataFrame):
		"""
		Drop the column 'threesixty' from the dataframe
		"""
		df.drop('threesixty', axis=1, inplace=True)
		return
	
	def dvn_transformation(df: pd.DataFrame):
		"""
		Drop the column 'dvn' from the dataframe
		"""
		df.drop('dvn', axis=1, inplace=True)
		return
	
	def discountValue_transformation(df: pd.DataFrame):
		"""
		Drop the column 'discountValue' from the dataframe
		"""
		df.drop('discountValue', axis=1, inplace=True)
		return
	
	def carType_transformation(df: pd.DataFrame):
		"""
		Drop the column 'carType' from the dataframe
		"""
		df.drop('carType', axis=1, inplace=True)
		return
	
	def NumOfCylinder_transformation(df: pd.DataFrame):
		"""
		Replace the value of No of Cylinder with null if the car is electric
		"""
		df.loc[(df['fuel'] == 'Electric') & (
			df['No of Cylinder'].notnull()), 'No of Cylinder'] = np.nan
		return
	
	def Height_transformation(df: pd.DataFrame):
		"""
		Drop the 'Height' column
		"""
		df.drop('Height', axis=1, inplace=True)
		return
	
	def Length_transformation(df: pd.DataFrame):
		"""
		Drop the 'Length' column
		"""
		df.drop('Length', axis=1, inplace=True)
		return
	
	def RearTread_transformation(df: pd.DataFrame):
		"""
		Drop the 'Rear Tread' column
		"""
		df.drop('Rear Tread', axis=1, inplace=True)
		return
	
	def GrossWeight_transformation(df: pd.DataFrame):
		"""
		Drop the 'Gross Weight' column
		"""
		df.drop('Gross Weight', axis=1, inplace=True)
		return
	
	def Seats_transformation(df: pd.DataFrame):
		"""
		Replace the Seats with null if the value is zero
		"""
		df.loc[(df['Seats'] == 0), 'Seats'] = np.nan
		return
	
	def TurningRadius_transformation(df: pd.DataFrame):
		"""
		Replace the Turning Radius with null if the value is greater and 15
		"""
		df.loc[(df['Turning Radius'] > 15.0), 'Turning Radius'] = np.nan
		return
	
	def Doors_transformation(df: pd.DataFrame):
		"""
		Replace all rows having Doors=5 with Doors=4
		"""
		df.loc[(df['Doors'] == 5), 'Doors'] = 4
		return
	
	def model_type_new_transformation(df: pd.DataFrame):
		"""
		Drop the 'model_type_new' column
		"""
		df.drop('model_type_new', axis=1, inplace=True)
		return
	
	def exterior_color_transformation(df: pd.DataFrame):
		"""
		Drop the 'exterior_color' column
		"""
		df.drop('exterior_color', axis=1, inplace=True)
		return
	
	def GroundClearanceUnladen_transformation(df: pd.DataFrame):
		"""
		Drop the 'Ground Clearance Unladen' column
		"""
		df.drop('Ground Clearance Unladen', axis=1, inplace=True)
		return
	
	def CompressionRatio_transformation(df: pd.DataFrame):
		"""
		Drop the 'Compression Ratio' column
		"""
		df.drop('Compression Ratio', axis=1, inplace=True)
		return
	
	def AlloyWheelSize_transformation(df: pd.DataFrame):
		"""
		Replace the rows having Alloy Wheel Size=7 with null
		"""
		df.loc[(df['Alloy Wheel Size'] == 7), 'Alloy Wheel Size'] = np.nan
		return
	
	def MaxTorqueAt_transformation(df: pd.DataFrame):
		"""
		Mark all the rows having Max Torque At below 1000 and above 5000 as null
		"""
		df.loc[(df['Max Torque At'] < 1000) | (
				df['Max Torque At'] > 5000), 'Max Torque At'] = np.nan
		return
	
	def Bore_transformation(df: pd.DataFrame):
		"""
		Drop the 'Bore' column
		"""
		df.drop('Bore', axis=1, inplace=True)
		return
	
	def Stroke_transformation(df: pd.DataFrame):
		"""
		Drop the 'Stroke' column
		"""
		df.drop('Stroke', axis=1, inplace=True)
		return
	
	def TARGET_transformation(df: pd.DataFrame):
		"""
		Drop the cars where the price is greater than 2_00_00_000
		"""
		df.drop(df[df[TARGET] > 2_00_00_000].index, inplace=True)
		return


def run_transformations(
		filepath: str = None,
		df: pd.DataFrame = None,
		save_to_file: bool = True
) -> str | pd.DataFrame:
	if df is None:
		if filepath is None:
			dl = DataLoader(dir_path=CLEAN_DIR_PATH)
			file_name = dl.get_latest_file(begins_with=CLEAN_FILE_BEGIN)
			filepath = os.path.join(CLEAN_DIR_PATH, file_name)
		df = pd.read_csv(filepath, index_col=INDEX)
	
	# Get all the functions from the Transformations class
	suggested_transformations = [func for func in dir(Transformations) if callable(
		getattr(Transformations, func)) and not func.startswith("__")]
	
	# Apply all the suggested transformations
	for transformation in suggested_transformations:
		getattr(Transformations, transformation)(df)
	# print(transformation)
	
	if not save_to_file:
		return df
	
	# Save the transformed data
	save_filename = f"{PROCESSED_FILE_BEGIN}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
	save_filepath = os.path.join(PROCESSED_DIR_PATH, save_filename)
	df.to_csv(save_filepath)
	
	return save_filepath


def main():
	print('Running transformations...')
	file_path = run_transformations()
	print(f"Transformed data saved to {file_path}")


if __name__ == '__main__':
	main()
