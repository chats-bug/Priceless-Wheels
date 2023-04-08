from catboost import CatBoostRegressor
import pandas as pd
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering.feature_transformations import FeatureEngineeringTransformations
from src.utils.constants import TARGET


class CatBoostModel:
	def __init__(self):
		self.num_features_after_transformation = None
		self.cat_features_after_transformation = None
		self.preprocessor = None
		self.model = CatBoostRegressor(
			learning_rate=0.08,
			depth=5,
			l2_leaf_reg=4,
			loss_function='MAE',
			bootstrap_type='Bernoulli',
			subsample=0.5,
			random_seed=42,
			allow_writing_files=False,
			verbose=100,
			n_estimators=5000,
			early_stopping_rounds=200,
		)
	
	def get_preprocessor(self, X: pd.DataFrame) -> pipeline:
		X = X.copy()
		numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
		categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
		nan_cols = X.columns[X.isna().any()].tolist()
		
		# TODO: Add the matching imputation
		# Matching imputation : A custom imputation which aggregates the data by the columns specified in the
		# arguments and gets an exact match if available.
		# for col in nan_cols:
		# 	args = MatchingImputerArguments(
		# 		strategy=pd.Series.mode if col in categorical_cols else pd.Series.mean,
		# 		columns=['model', 'variant'],
		# 		add_indicator=True,
		# 		copy=False,
		# 		match_level=0,
		# 	)
		# 	matching_imputer = CustomMatchingImputer(target=col, imputer_arguments=args)
		# 	X[col] = matching_imputer.fit_transform(X)
		
		# Feature engineering transformations and changing of columns accordingly
		feature_engineering = FeatureEngineeringTransformations(target=TARGET)
		replace_col_scores = feature_engineering.object_cols
		replacement_col_scores = [f'{col}_score' for col in replace_col_scores]
		
		# Replace the columns with the new columns
		# Remove the replace_col_scores from the categorical_cols
		categorical_cols = [col for col in categorical_cols if col not in replace_col_scores]
		# Add the replacement_col_scores to the categorical_cols
		numerical_cols.extend(replacement_col_scores)
		
		# Create a numerical transformer
		numerical_transformer = pipeline.Pipeline(
			steps=[
				('scaler', MinMaxScaler()),
			]
		)
		
		# Create a categorical transformer
		categorical_transformer = pipeline.Pipeline(
			steps=[
				('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
			]
		)
		
		# Create a column transformer
		column_transformer = ColumnTransformer(
			transformers=[
				('num', numerical_transformer, numerical_cols),
				('cat', categorical_transformer, categorical_cols)
			]
		)
		
		preprocessor = pipeline.Pipeline(
			steps=[
				('feature_engineering', feature_engineering),
				('column_transformer', column_transformer)
			]
		)
		
		self.preprocessor = preprocessor
		self.cat_features_after_transformation = [f'cat__{col}' for col in categorical_cols]
		self.num_features_after_transformation = [f'num__{col}' for col in numerical_cols]
		return preprocessor
	
	def get_model(self) -> CatBoostRegressor:
		return self.model
