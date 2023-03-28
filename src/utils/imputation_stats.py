from cmath import inf
from dataclasses import dataclass

import numpy as np


@dataclass
class KNNImputationArguments:
    n_neighbors: int = 5
    weights: str = 'uniform'
    metric: str = 'nan_euclidean'
    copy: bool = True
    add_indicator: bool = True
    missing_values = np.nan
    n_jobs: int | None = None
    keep_empty_features: bool = False
    columns: list[str] = None


@dataclass
class IterativeImputationArguments:
    estimator = None
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
    columns: list[str] = None


@dataclass
class SimpleImputerArguments:
    strategy: str = 'mean'
    fill_value = None
    missing_values = np.nan
    verbose: int = 0
    copy: bool = True
    add_indicator: bool = True
    keep_empty_features: bool = False


imp = SimpleImputerArguments()
# Convert to a dictionary
imp_dict = imp.__dict__
# Remove the columns key
print(imp_dict.pop('columns') if 'columns' in imp_dict else None)
print(imp_dict)