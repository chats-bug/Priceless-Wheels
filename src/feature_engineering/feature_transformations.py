from ast import literal_eval
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

from src.utils.constants import INDEX, TARGET
from src.utils.data_loader import DataLoader


geolocator = Nominatim(user_agent="geoapiExercises")


class FeatureEngineeringTransformations:
	"""
	This class contains all the recommended feature engineeing transformations for the dataset.
	"""

	def __init__(
		self, 
		df: pd.DataFrame, 
		object_cols: list[str] = ['top_features', 'comfort_features','interior_features', 'exterior_features', 'safety_features']
	):
		self.df = df
		self.object_cols = object_cols
		self.feature_prices = None
		self.is_fitted = False

	def _car_object_feature_dict(self) -> dict:
		unique_feature_scores = dict()
		for col in self.object_cols:
			for _, row in self.df.iterrows():
				feature_list = literal_eval(row[col])
				for feature in feature_list:
					if feature in unique_feature_scores.keys():
						unique_feature_scores[feature][1] += 1
						unique_feature_scores[feature][0] += row[TARGET]
					else:
						unique_feature_scores[feature] = [row[TARGET], 1]

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

			# Loop throught the rows with zero score and find a better score
			# Most probably cause: the feature set is empty, not provided by the user
			for _, zero_row in df[df[f'{col}_score'] == 0].iterrows():
				zero_oem = zero_row['oem']
				zero_model = zero_row['model']
				zero_year = zero_row['myear']
				zero_variant = zero_row['variant']

				print(f'Zero score for {col} for {zero_oem} {zero_model} {zero_year} {zero_variant}')

				# First check if the oem, model, year, variant are the same
				condition = (self.df['oem'] == zero_oem) & (self.df['model'] == zero_model) & (self.df['myear'] == zero_year) & (self.df['variant'] == zero_variant) 
				zero_row[f'{col}_score'] = self.df[condition][f'{col}_score'].mean()
				# Update the score in the original dataframe
				df.loc[zero_row.name, f'{col}_score'] = zero_row[f'{col}_score']
				if zero_row[f'{col}_score'] != 0 and zero_row[f'{col}_score'] is not np.nan:
					print(f'First check triggered {col} -> {zero_row[f"{col}_score"]}')
					continue

				# Second check if the oem, model, variant are the same
				condition = (self.df['oem'] == zero_oem) & (self.df['model'] == zero_model) & (self.df['variant'] == zero_variant)
				zero_row[f'{col}_score'] = self.df[condition][f'{col}_score'].mean()
				# Update the score in the original dataframe
				df.loc[zero_row.name, f'{col}_score'] = zero_row[f'{col}_score']
				if zero_row[f'{col}_score'] != 0 and zero_row[f'{col}_score'] is not np.nan:
					print(f'Second triggered {col} -> {zero_row[f"{col}_score"]}')
					continue

				# Third check if the oem, model, year are the same
				condition = (self.df['oem'] == zero_oem) & (self.df['model'] == zero_model) & (self.df['myear'] == zero_year)
				zero_row[f'{col}_score'] = self.df[condition][f'{col}_score'].mean()
				# Update the score in the original dataframe
				df.loc[zero_row.name, f'{col}_score'] = zero_row[f'{col}_score']
				if zero_row[f'{col}_score'] != 0 and zero_row[f'{col}_score'] is not np.nan:
					print(f'Third check triggered {col} -> {zero_row[f"{col}_score"]}')
					continue

				# Fourth check if the oem, model are the same
				condition = (self.df['oem'] == zero_oem) & (self.df['model'] == zero_model)
				zero_row[f'{col}_score'] = self.df[condition][f'{col}_score'].mean()
				# Update the score in the original dataframe
				df.loc[zero_row.name, f'{col}_score'] = zero_row[f'{col}_score']
				if zero_row[f'{col}_score'] != 0 and zero_row[f'{col}_score'] is not np.nan:
					print(f'Fourth check triggered {col} -> {zero_row[f"{col}_score"]}')
					continue

				# Fifth check if the oem, year are the same
				condition = (self.df['oem'] == zero_oem) & (self.df['myear'] == zero_year)
				zero_row[f'{col}_score'] = self.df[condition][f'{col}_score'].mean()
				# Update the score in the original dataframe
				df.loc[zero_row.name, f'{col}_score'] = zero_row[f'{col}_score']
				if zero_row[f'{col}_score'] != 0 and zero_row[f'{col}_score'] is not np.nan:
					print(f'Fifth check triggered {col} -> {zero_row[f"{col}_score"]}')
					continue

				# Sixth check if the oem is the same
				condition = (self.df['oem'] == zero_oem)
				zero_row[f'{col}_score'] = self.df[condition][f'{col}_score'].mean()
				# Update the score in the original dataframe
				df.loc[zero_row.name, f'{col}_score'] = zero_row[f'{col}_score']
				if zero_row[f'{col}_score'] != 0 and zero_row[f'{col}_score'] is not np.nan:
					print(f'Sixth check triggered {col} -> {zero_row[f"{col}_score"]}')
					continue
		
		return df

	def _get_lat_lon(self, row: pd.Series) -> str:
		location = geolocator.geocode(f'{row["loc"]} {row["City"]} {row["state"]}')
		if location is None:
			location = geolocator.geocode(f'{row["City"]} {row["state"]}')
		(lat, lon) = (location.raw['display_name']['lat'], location.raw['display_name']['lon']) if location else (np.nan, np.nan)
		return f'{lat},{lon}'

	def _location_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
		# Use geopy to get the zip code from the location: 'loc' + 'City' + 'state' -> 'zip'
		df['lat_lon'] = df.apply(self._get_lat_lon, axis=1)
		df['lat'] = df['lat_lon'].apply(lambda x: x.split(',')[0])
		df['lon'] = df['lat_lon'].apply(lambda x: x.split(',')[1])
		df.drop('lat_lon', axis=1, inplace=True)
		return df

	def fit(self):
		self.feature_prices = self._car_object_feature_dict()
		self.df = self._car_object_feature_transformation(self.df)
		self.is_fitted = True
		return self

	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		if self.is_fitted == False:
			raise Exception('Please fit the transformer first')
		# Transform the object columns to scores
		df = self._car_object_feature_transformation(df)
		df.drop(self.object_cols, axis=1, inplace=True)

		# Transform the location to zip code
		# df = self._location_transformation(df)
		# We temporarily suspend fetching the zip code from the location because it takes too long and the result isn't accurate enough
		
		return df