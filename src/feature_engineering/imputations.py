from __future__ import annotations

from category_encoders import OneHotEncoder, TargetEncoder

import numpy as np
from src.utils.constants import INDEX, ImputationStrategy, TARGET
from src.utils.imputation_stats import KNNImputerArguments, IterativeImputerArguments, MatchingImputerArguments

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR_PATH = '../../data/processed/'


class CustomMatchingImputer(BaseEstimator, TransformerMixin):
	"""
	A custom imputer which imputes missing values in a column by matching the closest value in another column manually.
	- We aggregate the values in the target column by the columns specified in the match_by parameter.
	- We then find the closest match for each value in the target column.
	- This imputer doesn't guarantee that all values in the target column will be imputed.

	Parameters
	-------
		target : str
			The name of the column to impute
		imputer_arguments : MatchingImputer
			The imputer_arguments to be used for imputation using KNNImputer
	
	Returns
	-------
	X : array-like
		The array with imputed values in the target column
	"""
	
	def __init__(
			self,
			target: str | None = None,
			imputer_arguments: MatchingImputerArguments = MatchingImputerArguments(columns=['model', 'variant']),
	) -> None:
		super().__init__()
		assert isinstance(imputer_arguments, MatchingImputerArguments) == True, \
			'Unrecognized value for imputer_arguments, should be MatchingImputer object'
		assert isinstance(target, str) or target is None, 'target should be a string or None'
		
		self.target = target
		self.imputer_arguments = imputer_arguments
		self.match_by = imputer_arguments.columns
		self.strategy = imputer_arguments.strategy
		self.missing_values = imputer_arguments.missing_values
		self.add_indicator = imputer_arguments.add_indicator
		self.copy = imputer_arguments.copy
		self.match_level = imputer_arguments.match_level
		self.match_level_array = np.array(imputer_arguments.match_level_array) or np.array(
			[0] * self.match_level + [1] * (len(self.match_by) - self.match_level)
		)
		self.car_groups = None
		
		assert self.match_level >= 0, 'match_level should be greater than or equal to 0'
		assert self.match_level < len(
			self.match_by), 'match_level should be less than the number of columns in match_by'
		assert len(self.match_by) == len(
			self.match_level_array), 'match_by and match_level_array should be of the same length'
		assert np.all(np.isin(self.match_level_array, [0, 1])) and self.match_level_array.dtype == int, \
			'match_level_array should be an array of integers with values 0 or 1'
	
	def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CustomMatchingImputer':
		"""
		Fits the imputer on the dataset

		Parameters
		----------
		X : pd.DataFrame
			The dataset to fit the imputer on
		y : pd.Series
			The target column

		Returns
		-------
		self : CustomMatchingImputer
			The fitted imputer
		"""
		self._validate_input(X)
		if self.target is None:
			if y is None:
				raise ValueError('target cannot be None if y is None')
			else:
				if self.target is None:
					self.target = y.name.__str__()
					X[self.target] = y
		
		self.match_by = self.match_by or X.columns.drop(self.target)
		self.match_by = [col for col in self.match_by if col != self.target]
		self.car_groups = X.groupby(self.match_by)[self.target].agg(self.strategy).reset_index(drop=False)
		return self
	
	def _get_closest_car_match(self, X: pd.DataFrame) -> pd.Series:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataset to impute

		Returns
		-------
		closest_match : pd.Series
			The closest match for each row in the dataset
		"""
		# Get the closest match for the row
		# If the value is missing, we try to find the closest match
		# This is how we find a match:
		# First we try to match every col in the self.match_by list to the row
		# If we find a match, we use the value of the closest match
		# If we don't find a match, we drop the last col in the self.match_by list and try again
		# Repeat until we find a match or we run out of cols to match
		# There is no guarantee that we will find a match
		X = X.copy()
		missing_rows = X[(X[self.target].isna()) | (X[self.target].isin(self.missing_values))]
		for index, row in missing_rows.iterrows():
			closest_match = np.nan
			condition = True
			for i, col in enumerate(self.match_by):
				condition = condition & (self.car_groups[col] == row[col])
				if self.match_level_array[i] == 1:
					closest_match = self.car_groups[condition][self.target].agg(self.strategy)
					if np.isnan(closest_match) or closest_match in self.missing_values:
						continue
					X.loc[index, self.target] = closest_match
		
		return X
	
	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		"""
		Transforms the dataset by imputing the missing values in the target column

		Parameters
		----------
		X : pd.DataFrame
			The dataset to transform

		Returns
		-------
		X : pd.DataFrame
			The transformed dataset
		"""
		check_is_fitted(self, 'car_groups')
		self._validate_input(X)
		
		# Copy the dataset if needed
		if self.copy:
			X = X.copy()
		
		# Add an indicator column if needed
		if self.add_indicator:
			X[f'{self.target}_imputed'] = X[self.target].isin(self.missing_values).astype(int)
		# Impute the missing values in the target column
		# X = X.apply(self._get_closest_car_match, axis=1)
		X = self._get_closest_car_match(X)
		return X
	
	def _validate_input(self, X: pd.DataFrame) -> None:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to validate

		Returns
		-------
		None
		"""
		if not isinstance(X, pd.DataFrame):
			raise ValueError("X should be a pandas dataframe")
	
	def _validate_target(self, X: pd.DataFrame, y: pd.Series) -> None:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to validate

		Returns
		-------
		None
		"""
		if y is None:
			if self.target is None:
				raise ValueError("Target column name not provided")
			y = X[self.target]
		else:
			self.target = y.name
		if y.name not in X.columns:
			raise ValueError(f"Target column {self.target} not found in dataframe")


class CustomKNNImputer(BaseEstimator, TransformerMixin):
	"""
	Custom Simple Imputer class for imputing missing values in the dataset. This imputer works in the following way:
	- We first look for the closest match in the dataset manually. If we find a match, we use the value of the closest match.
	- If we don't find a match, we use a SimpleImputer with the strategy specified in the constructor.
	
	This has some extra features compared to the sklearn KNNImputer:
	- We can specify a list of columns to group by. This is useful when we want to impute a column based on the values of other columns.
	- We allow categorical columns to be imputed and used for imputation as well. This is done by label encoding the categorical columns and then imputing the numerical columns. After imputation, we reverse the label encoding to convert the columns back to categorical.
	
	Suggestions for improvement:
	- If we want to use categorical columns for imputation, we should use a different strategy than label encoding. One-hot encoding is a good option.
	- If the cardinality of the column is too high to use one-hot encoding, we can also use a target encoder.
	- Make sure to standardize the numerical columns passing them to the imputer.
	
	Parameters
	----------
		target : str
			The name of the column to impute
		imputer_arguments : KNNImputerArguments
			The imputer_arguments to be used for replacement using KNNImputer
		encoding : str
			The encoding to use for categorical columns. Can be 'onehot' or 'label' or 'target' (not implemented yet)
	
	Returns
	-------
		X : array-like
			The array with imputed values in the target column
	"""
	
	def __init__(
			self,
			target: str = None,
			imputer_arguments: KNNImputerArguments = KNNImputerArguments(),
			encoding: str = 'onehot',
	) -> None:
		assert isinstance(imputer_arguments, KNNImputerArguments) == True, \
			'Unrecognized value for imputer_arguments, should be SimpleImputerArguments object'
		
		super().__init__()
		self.target = target
		self.imputer_arguments = imputer_arguments.__dict__
		self.imputer_arguments = self.imputer_arguments.copy()
		self.group_cols = self.imputer_arguments.pop('columns')
		if self.group_cols is not None and self.target in self.group_cols:
			self.group_cols.remove(self.target)
		self.encoding = encoding
		self.encoders = {}
		self.encoded_cols = []
		
		self.copy = self.imputer_arguments['copy']
		self.imputer_arguments['copy'] = False
	
	def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CustomKNNImputer':
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to fit the imputer on
		y : pd.Series
			The target column to impute

		Returns
		-------
		self : CustomKNNImputer
			The fitted imputer
		"""
		self._validate_input(X)
		self._validate_target(X, y)
		
		X = X.copy()
		
		self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
		self.group_cols = self.group_cols if self.group_cols else self.numerical_cols
		if self.target not in self.group_cols:
			self.group_cols.append(self.target)
		
		# Encode the categorical columns
		for col in self.group_cols:
			if X[col].dtype.name in ['category', 'object']:
				X = self._encode_column(X, col)
		
		self.group_cols.extend([self.target])
		self.imputer = KNNImputer(**self.imputer_arguments)
		self.imputer.fit(X[self.group_cols].values)
		
		return self
	
	def _encode_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to encode
		col : str
			The column to encode

		Returns
		-------
		X : pd.DataFrame
			The encoded dataframe
		"""
		X[col] = X[col].astype('category')
		
		if self.encoding == 'label' or self.target == col:
			if self.encoders.get(col) is None:
				self.encoders[col] = LabelEncoder()
				self.encoded_cols.append(col)
			X[col] = self.encoders[col].fit_transform(X[col])
		elif self.encoding == 'onehot':
			if self.encoders.get(col) is None:
				self.encoders[col] = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
				self.encoded_cols.append(col)
			X[col] = self.encoders[col].fit_transform(X[col].values.reshape(-1, 1))
		
		return X
	
	def _decode_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to decode
		col : str
			The column to decode

		Returns
		-------
		X : pd.DataFrame
			The decoded dataframe
		"""
		if col not in self.encoded_cols:
			return X
		
		if self.encoding == 'target' or self.encoding == 'label' or col == self.target:
			X[col] = self.encoders[col].inverse_transform(X[col].values.reshape(-1, 1))
		elif self.encoding == 'onehot':
			print(X.columns)
			one_hot_cols = [c for c in X.columns if c.startswith(f'{col}_')]
			print(f'{col} onehot columns : {one_hot_cols}')
			t = self.encoders[col].inverse_transform(X[one_hot_cols])
			print(f'{col} : {t.shape}')
			print(f'{col} : {X[one_hot_cols].shape}')
			# print(t)
			X[col] = self.encoders[col].inverse_transform(X[one_hot_cols])
			X = X.drop([col for col in X.columns if col.startswith(f'{col}_')], axis=1)
		
		return X
	
	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to transform

		Returns
		-------
		X : pd.DataFrame
			The transformed dataframe
		"""
		self._validate_input(X)
		
		check_is_fitted(self, 'imputer')
		
		if self.copy:
			X = X.copy()
		
		# print(f'{X.isna().sum()}')
		# Encode categorical columns
		for col in self.encoded_cols:
			X = self._encode_column(X, col)
		
		# Impute missing values
		self.imputer.transform(X[self.group_cols])
		
		# Decode categorical columns
		for col in self.encoded_cols:
			X = self._decode_column(X, col)
		
		# print(f'{X.isna().sum()}')
		return X
	
	def _validate_input(self, X: pd.DataFrame) -> None:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to validate

		Returns
		-------
		None
		"""
		if not isinstance(X, pd.DataFrame):
			raise ValueError("X should be a pandas dataframe")
	
	def _validate_target(self, X: pd.DataFrame, y: pd.Series) -> None:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to validate

		Returns
		-------
		None
		"""
		if y is None:
			if self.target is None:
				raise ValueError("Target column name not provided")
			y = X[self.target]
		else:
			self.target = y.name
		if y.name not in X.columns:
			raise ValueError(f"Target column {self.target} not found in dataframe")


class CustomIterativeImputer(BaseEstimator, TransformerMixin):
	"""
	Custom Simple Imputer class for imputing missing values in the dataset. This has some extra features compared to the sklearn IterativeImputer:
	- We can specify a list of columns to group by. This is useful when we want to impute a column based on the values of other columns.
	- We allow categorical columns to be imputed and used for imputation as well. This is done by label encoding the categorical columns and then imputing the numerical columns. After imputation, we reverse the label encoding to convert the columns back to categorical.
	
	Suggestions for improvement:
	- If we want to use categorical columns for imputation, we should use a different strategy than label encoding. One-hot encoding is a good option.
	- If the cardinality of the column is too high to use one-hot encoding, we can use a hashing trick to reduce the cardinality.
	- Or if we are not comfortable hashing, simply choose a non-linear estimator like a RandomForestRegressor to impute the missing values.
	
	Parameters
	----------
		imputer_arguments : IterativeImputerArguments
			The imputer_arguments to be used for replacement using the IterativeImputer
		encoding : str
			The strategy to use for encoding the categorical columns. Can be 'label' or 'onehot'. Defaults to 'label'
	
	Returns
	-------
		X : array-like
			The array with imputed values in the target column
	"""
	
	def __init__(
			self,
			imputer_arguments: IterativeImputerArguments = IterativeImputerArguments(),
			encoding: str = 'label',
	):
		assert isinstance(imputer_arguments, IterativeImputerArguments) == True, \
			'Unrecognized value for imputer_arguments, should be IterativeImputerArguments object'
		assert encoding in ['label', 'onehot'], 'encoding should be either label or onehot'
		
		self.imputer_arguments: dict = imputer_arguments.__dict__
		self.group_cols: list[str] = self.imputer_arguments.pop('columns')
		self.copy: bool = self.imputer_arguments.pop('copy')
		self.add_indicator: bool = self.imputer_arguments.pop('add_indicator')
		
		self.encoding = encoding
		self.encoders = {}
		self.encoding_mappings = {}
		self.scaler = StandardScaler()
	
	def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CustomIterativeImputer':
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to fit the imputer on
		y : pd.Series
			The target column

		Returns
		-------
		self : CustomIterativeImputer
			The fitted imputer
		"""
		self._validate_input(X)
		self._validate_target(X, y)
		
		X = X.copy()
		
		# Set the group columns to the columns specified in the imputer_arguments.
		# If the group columns are not specified, we use all the numeric columns except the target column
		self.group_cols = self.group_cols or [col for col in X.columns if X[col].dtype != 'object']
		
		self.encoded_cols = []
		for col in self.group_cols:
			if col not in X.columns:
				raise ValueError(f"Column {col} not found in dataframe")
			
			if X[col].dtype == 'object':
				self.encoded_cols.append(col)
				X = self._encode_column(X, col)
		
		# Fit the scalar and transform the data
		# Only fit and transform the group columns which are not one-hot encoded
		if self.encoding == 'onehot':
			non_encoded_cols = [col for col in self.group_cols if col not in self.encoded_cols]
			self.scaler.fit(X[non_encoded_cols])
			X[non_encoded_cols] = self.scaler.transform(X[non_encoded_cols])
		else:
			self.scaler.fit(X[self.group_cols])
			X[self.group_cols] = self.scaler.transform(X[self.group_cols])
		
		self.imputer = IterativeImputer(**self.imputer_arguments)
		self.imputer.fit(X[self.group_cols])
		return self
	
	def _encode_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to encode
		col : str
			The column to encode

		Returns
		-------
		X : pd.DataFrame
			The encoded dataframe
		"""
		X[col] = X[col].astype('category')
		
		if self.encoding == 'label':
			if self.encoders.get(col) is None:
				self.encoders[col] = LabelEncoder()
			X[col] = self.encoders[col].fit_transform(X[col])
		elif self.encoding == 'target':
			if self.encoders.get(col) is None:
				self.encoders[col] = TargetEncoder()
			X[col] = self.encoders[col].fit_transform(X[col], TARGET)
		elif self.encoding == 'onehot':
			if self.encoders.get(col) is None:
				self.encoders[col] = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
			X[col] = self.encoders[col].fit_transform(X[col].values.reshape(-1, 1))
		return X
	
	def _decode_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to decode
		col : str
			The column to decode

		Returns
		-------
		X : pd.DataFrame
			The decoded dataframe
		"""
		if col not in self.encoded_cols:
			return X
		
		if self.encoding == 'label':
			X[col] = self.encoders[col].inverse_transform(X[col].values.reshape(-1, 1))
		elif self.encoding == 'onehot':
			one_hot_cols = [c for c in X.columns if c.startswith(f'{col}_')]
			X[col] = self.encoders[col].inverse_transform(X[one_hot_cols])
			X = X.drop([col for col in X.columns if col.startswith(f'{col}_')], axis=1)
		
		return X
	
	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to transform

		Returns
		-------
		X : pd.DataFrame
			The transformed dataframe
		"""
		self._validate_input(X)
		
		# Make a copy if we are not allowed to modify the original dataframe
		if self.copy:
			X = X.copy()
		
		# Before we transform, we need to make sure that the categorical columns are encoded
		# Encode the columns
		for col in self.group_cols:
			if col not in X.columns:
				raise ValueError(f"Column {col} not found in dataframe")
			
			if X[col].dtype == 'object':
				X = self._encode_column(X, col)
		
		# Standardize the dataframe
		# Only transform the group columns which are not one-hot encoded
		if self.encoding == 'onehot':
			non_encoded_cols = [col for col in self.group_cols if col not in self.encoded_cols]
			X[non_encoded_cols] = self.scaler.transform(X[non_encoded_cols])
		else:
			X[self.group_cols] = self.scaler.transform(X[self.group_cols])
		
		# Transform the dataframe
		imputed = self.imputer.transform(X[self.group_cols])
		X[self.group_cols] = imputed
		
		# De-standardize the dataframe
		# Only transform the group columns which are not one-hot encoded
		if self.encoding == 'onehot':
			non_encoded_cols = [col for col in self.group_cols if col not in self.encoded_cols]
			X[non_encoded_cols] = self.scaler.inverse_transform(X[non_encoded_cols])
		else:
			X[self.group_cols] = self.scaler.inverse_transform(X[self.group_cols])
		
		# Decode the columns
		for col in self.encoded_cols:
			X = self._decode_column(X, col)
		return X
	
	def _validate_input(self, X: pd.DataFrame) -> None:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to validate

		Returns
		-------
		None
		"""
		if not isinstance(X, pd.DataFrame):
			raise ValueError("X should be a pandas dataframe")
	
	def _validate_target(self, X: pd.DataFrame, y: pd.Series) -> None:
		"""
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to validate

		Returns
		-------
		None
		"""
		if y is None:
			if self.target is None:
				raise ValueError("Target column name not provided")
			y = X[self.target]
		else:
			self.target = y.name
		if y.name not in X.columns:
			raise ValueError(f"Target column {self.target} not found in dataframe")
