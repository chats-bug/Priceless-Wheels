import os

import numpy as np
from src.utils import data_loader
from src.utils.constants import INDEX, ImputationStragety
from src.utils.imputation_stats import KNNImputationArguments, IterativeImputationArguments, SimpleImputerArguments

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

PROCESSED_DIR_PATH = '../../data/processed/'


class CustomImputer(BaseEstimator):
    """
    Custom imputer for missing values in the dataset. This imputer uses a combination of different imputation stratergies.
    - We first look for the closest match in the dataset manually. If we find a match, we use the value of the closest match.
    - If we don't find a match, we use one of the following

    """

    def __init__(self, stratergy: ImputationStragety = ImputationStragety.CUSTOM) -> None:
        if stratergy not in ImputationStragety:
            raise ValueError(f'Invalid imputation stratergy: {stratergy}')

        super().__init__()
        self.stratergy = stratergy

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the imputer on the dataset.
        """
        self.X = X
        self.y = y
        return self


class ColumnWiseImputations:
    """
    This class contains functions which perform imputations on speicific columns in the dataset.
    """

    def __init__(self, df: pd.DataFrame, match1: str = 'model', match2: str = 'variant') -> None:
        self.df = df
        self.match1 = match1
        self.match2 = match2

    def exact_car_match(self, column: str) -> pd.DataFrame:
        """
        Find the exact match for the given column in the dataset.
        """

        missing_mask = self.df[column].isna()
        missing_df = self.df[missing_mask]

        # Get a list of unique model-variant combinations in the missing dataframe
        unique_combinations = missing_df[[
            self.match1, self.match2]].drop_duplicates()

        # Loop through the unique combinations and fill in the missing values
        for model, variant in unique_combinations.itertuples(index=False):
            # Find rows in the original dataframe that match the current model and variant
            mask = (self.df[self.match1] == model) & (
                self.df[self.match2] == variant) & (self.df[column].notnull())
            matched_rows = self.df[mask]
            if not matched_rows.empty:
                # Fill in the missing value with the mean of the matched rows
                mean = matched_rows[column].mean()
                missing_mask = (missing_df[self.match1] == model) & (
                    missing_df[self.match2] == variant)
                missing_df.loc[missing_mask, column] = mean

        # Update the original dataframe with the imputed values
        self.df.update(missing_df)
        return self.df

    def car_model_match(self, column: str) -> pd.DataFrame:
        """
        Find the closest match for the given column in the dataset.
        """

        missing_mask = self.df[column].isna()
        missing_df = self.df[missing_mask]

        # Get a list of unique model-variant combinations in the missing dataframe
        unique_combinations = missing_df[[self.match1]].drop_duplicates()

        # Loop through the unique combinations and fill in the missing values
        for model in unique_combinations.itertuples(index=False):
            # Find rows in the original dataframe that match the current model and variant
            mask = (self.df[self.match1] == model) & (
                self.df[column].notnull())
            matched_rows = self.df[mask]
            if not matched_rows.empty:
                # Fill in the missing value with the mean of the matched rows
                mean = matched_rows[column].mean()
                missing_mask = (missing_df[self.match1] == model)
                missing_df.loc[missing_mask, column] = mean

        # Update the original dataframe with the imputed values
        self.df.update(missing_df)
        return self.df

    def numerical_imputations(
            self,
            columns: list[str] = None,
            stratergy: dict[str, ImputationStragety] = {
                'all': ImputationStragety.KNN},
            stratergy_kwargs: dict[str, object] = {
                'all': KNNImputationArguments()},
    ) -> pd.DataFrame:
        """
        Perform imputations on the given column.
        """
        # Get the numerical columns
        numerical_columns = self.df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        columns = columns or numerical_columns

        # Check if the given columns are present in numerical columns
        if set(columns) - set(numerical_columns):
            raise ValueError(
                f'Invalid columns: {set(columns) - set(numerical_columns)}')

        # First get the columns that have missing values
        missing_columns = [
            column for column in columns if self.df[column].isna().any()]

        # Check if the stratergy is valid for the columns
        for column in stratergy.keys():
            if column not in columns and column != 'all':
                raise ValueError(
                    f'Invalid column: {column} for stratergy: {stratergy[column]}')
            if stratergy[column] not in ImputationStragety:
                raise ValueError(
                    f'Invalid imputation stratergy: {stratergy[column]}')

        # Check if the stratergy kwargs are valid for the columns
        for column in stratergy_kwargs.keys():
            if column not in columns and column != 'all':
                raise ValueError(
                    f'Invalid column: {column} for stratergy kwargs: {stratergy_kwargs[column]}')

            # Check if the stratergy kwargs are valid for the stratergy
            match stratergy[column]:
                case ImputationStragety.KNN:
                    if not isinstance(stratergy_kwargs[column], KNNImputationArguments):
                        raise ValueError(
                            f'Invalid stratergy kwargs for KNN imputation: {stratergy_kwargs[column]}')
                case ImputationStragety.MEAN, ImputationStragety.MEDIAN, ImputationStragety.MODE, ImputationStragety.CONSTANT:
                    if not isinstance(stratergy_kwargs[column], SimpleImputerArguments):
                        raise ValueError(
                            f'Invalid stratergy kwargs for constant imputation: {stratergy_kwargs[column]}')
                    if stratergy[column] == ImputationStragety.CONSTANT and (stratergy_kwargs[column].stratergy != 'constant' or stratergy_kwargs[column].fill_value is None):
                        raise ValueError(
                            f'Invalid stratergy kwargs for constant imputation: {stratergy_kwargs[column]}')
                    elif stratergy[column] == ImputationStragety.MEAN and stratergy_kwargs[column].stratergy != 'mean':
                        raise ValueError(
                            f'Invalid stratergy kwargs for mean imputation: {stratergy_kwargs[column]}')
                    elif stratergy[column] == ImputationStragety.MEDIAN and stratergy_kwargs[column].stratergy != 'median':
                        raise ValueError(
                            f'Invalid stratergy kwargs for median imputation: {stratergy_kwargs[column]}')
                    elif stratergy[column] == ImputationStragety.MODE and stratergy_kwargs[column].stratergy != 'most_frequent':
                        raise ValueError(
                            f'Invalid stratergy kwargs for mode imputation: {stratergy_kwargs[column]}')

                case ImputationStragety.MICE:
                    if not isinstance(stratergy_kwargs[column], IterativeImputationArguments):
                        raise ValueError(
                            f'Invalid stratergy kwargs for MICE imputation: {stratergy_kwargs[column]}')

        # Loop through the missing columns and perform imputations
        for column in missing_columns:
            # Step 1: Exact match imputations
            self.df = self.exact_car_match(column)
            # Step 2: Car model match imputations
            self.df = self.car_model_match(column)

            matched_stratergy = stratergy[column] if column in stratergy else stratergy['all']
            matched_stratergy_kwargs = stratergy_kwargs[matched_stratergy].__dict__
            columns_to_fit = matched_stratergy_kwargs.pop(
                'columns') if 'columns' in matched_stratergy_kwargs else numerical_columns

            # Get the strategy for imputing the missing values
            match matched_stratergy:
                case ImputationStragety.MEAN:
                    imputer = SimpleImputer(
                        matched_stratergy_kwargs)
                    self.df[column] = imputer.fit_transform(
                        self.df[column].values.reshape(-1, 1))

                case ImputationStragety.MEDIAN:
                    imputer = SimpleImputer(
                        matched_stratergy_kwargs)
                    self.df[column] = imputer.fit_transform(
                        self.df[column].values.reshape(-1, 1))

                case ImputationStragety.MODE:
                    imputer = SimpleImputer(
                        matched_stratergy_kwargs)
                    self.df[column] = imputer.fit_transform(
                        self.df[column].values.reshape(-1, 1))

                case ImputationStragety.CONSTANT:
                    imputer = SimpleImputer(
                        matched_stratergy_kwargs)
                    self.df[column] = imputer.fit_transform(
                        self.df[column].values.reshape(-1, 1))

                case ImputationStragety.KNN:
                    imputer = KNNImputer(
                        matched_stratergy_kwargs)
                    self.df[column] = imputer.fit_transform(
                        self.df[columns_to_fit].values.reshape(-1, 1))

                case ImputationStragety.MICE:
                    columns_to_use = stratergy_kwargs[column].columns or numerical_columns
                    columns_to_use = columns_to_use.append(
                        column) if column not in columns_to_use else columns_to_use
                    imputer = IterativeImputer(**stratergy_kwargs)
                    self.df[column] = imputer.fit_transform(
                        self.df[columns_to_fit].values.reshape(-1, 1))

                case _:
                    raise ValueError(
                        f'Invalid imputation stratergy: {stratergy}')

        return self.df

    def categorical_imputations(
            self,
            columns: list[str] = None,
            stratergy: dict[str, ImputationStragety] = {
                'all': ImputationStragety.CONSTANT},
            stratergy_kwargs: dict[str, object] = {
                'all': SimpleImputerArguments(strategy='most_frequent', fill_value='missing')},
    ) -> pd.DataFrame:
        """
        Perform imputations on the given categorical columns.
        """

        # Get the categorical columns
        categorical_columns = self.df.select_dtypes(
            include=['object']).columns.tolist()
        columns = columns or categorical_columns

        # Check if the given columns are present in categorical columns
        if set(columns) - set(categorical_columns):
            raise ValueError(
                f'Invalid columns: {set(columns) - set(categorical_columns)}')

        # First get the columns that have missing values
        missing_columns = [
            column for column in columns if self.df[column].isna().any()]

        # Check if the stratergy is valid for the columns
        for column in stratergy.keys():
            if column not in columns and column != 'all':
                raise ValueError(
                    f'Invalid column in stratergy: {column}')
            if stratergy[column] not in [ImputationStragety.MODE, ImputationStragety.CONSTANT]:
                raise ValueError(
                    f'Invalid imputation stratergy: {stratergy[column]}')

        # Check if the stratergy kwargs are valid for the columns
        for column in stratergy_kwargs.keys():
            if column not in columns and column != 'all':
                raise ValueError(
                    f'Invalid column in stratergy kwargs: {column}')

            # Check if the stratergy kwargs are valid for the stratergy
            match stratergy[column]:
                case ImputationStragety.CONSTANT:
                    if not isinstance(stratergy_kwargs[column], SimpleImputerArguments):
                        raise ValueError(
                            f'Invalid stratergy kwargs for constant imputation: {stratergy_kwargs[column]}')
                    if stratergy_kwargs[column].stratergy != 'constant' or stratergy_kwargs[column].fill_value == None:
                        raise ValueError(
                            f'Invalid stratergy kwargs for constant imputation: {stratergy_kwargs[column]}')

                case ImputationStragety.MODE:
                    if not isinstance(stratergy_kwargs[column], SimpleImputerArguments):
                        raise ValueError(
                            f'Invalid stratergy kwargs for mode imputation: {stratergy_kwargs[column]}')
                    if stratergy_kwargs[column].stratergy != 'most_frequent':
                        raise ValueError(
                            f'Invalid stratergy kwargs for mode imputation: {stratergy_kwargs[column]}')

                case _:
                    raise ValueError(
                        f'Invalid imputation stratergy: {stratergy}')

        # Loop through the missing columns and perform imputations
        for column in missing_columns:
            # Step 1: Exact match imputations
            self.df = self.exact_car_match(column)
            # Step 2: Car model match imputations
            self.df = self.car_model_match(column)

            matched_stratergy = stratergy[column] if column in stratergy else stratergy['all']
            matched_stratergy_kwargs = stratergy_kwargs[matched_stratergy].__dict__
            matched_stratergy_kwargs.pop('columns')

            # Get the strategy for imputing the missing values
            match matched_stratergy:
                case ImputationStragety.CONSTANT:
                    imputer = SimpleImputer(
                        matched_stratergy_kwargs)
                    self.df[column] = imputer.fit_transform(
                        self.df[column].values.reshape(-1, 1))
                case ImputationStragety.MODE:
                    imputer = SimpleImputer(
                        matched_stratergy_kwargs)
                    self.df[column] = imputer.fit_transform(
                        self.df[column].values.reshape(-1, 1))
                case _:
                    raise ValueError(
                        f'Invalid imputation stratergy: {stratergy}')
