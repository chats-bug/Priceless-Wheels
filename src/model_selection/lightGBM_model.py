from lightgbm import LGBMRegressor
import pandas as pd
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.feature_engineering.feature_transformations import FeatureEngineeringTransformations
from src.utils.constants import TARGET


class LightGBMModel:
	def __init__(self):
		self.model = LGBMRegressor(
			learning_rate=0.08363996779482333,
			num_leaves=26,
			max_depth=7,
			min_child_samples=14,
			subsample=0.8130687216963774,
			colsample_bytree=0.726149859230546,
			reg_alpha=6.495685321153756,
			reg_lambda=0.004206014748968054,
			n_estimators=1000,
			objective='regression',
			importance_type='gain',
			boosting_type='gbdt',
			verbose=1,
			min_split_gain=0.0,
			random_state=42,
			n_jobs=-1
		)
		self.preprocessor = None
	
	def get_preprocessor(self, X: pd.DataFrame) -> pipeline:
		X = X.copy()
		numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
		categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
		nan_cols = X.columns[X.isna().any()].tolist()
		
		# TODO: Add the matching imputation
		# Matching imputation : A custom imputation which aggregates the data by the columns specified in the
		# arguments and gets an exact match if available.
		# for col in nan_cols:
		# 	MatchingImputerArguments(
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
				('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
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
		return preprocessor

	def get_model(self):
		return self.model
	