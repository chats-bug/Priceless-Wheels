from cmath import inf
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import BayesianRidge


@dataclass
class MatchingImputerArguments:
	strategy: callable = pd.Series.mean  # The strategy to use for matching. Can be a str or any valid pandas aggregation function
	columns: list[str] = None  # The columns to match by. If None, all columns will be used
	missing_values = [np.nan, 'nan']  # The missing values to impute
	add_indicator: bool = True  # Whether to add an indicator column or not
	copy: bool = False  # Whether to copy the dataset or impute in place
	match_level: int = 0
	match_level_array: list[int] = None


@dataclass
class KNNImputerArguments:
	n_neighbors: int = 5
	weights: str = 'uniform'
	metric: str = 'nan_euclidean'
	copy: bool = True
	add_indicator: bool = True
	missing_values = np.nan
	keep_empty_features: bool = False
	columns: list[str] = None


@dataclass
class IterativeImputerArguments:
	estimator: BaseEstimator = BayesianRidge()
	sample_posterior: bool = False
	missing_values: int | float = np.nan
	initial_strategy: str = 'mean'
	imputation_order: str = 'ascending'
	max_iter: int = 10
	tol: float = 1e-3
	n_nearest_features = None
	verbose: int = 0
	random_state: int = None
	add_indicator: bool = True
	skip_complete: bool = False
	min_value = inf
	max_value = inf
	keep_empty_features = False
	copy: bool = True
	columns: list[str] = None  # Drop this before passing to sklearn imputer
