import numpy as np
import pandas as pd
from typing import List, Dict
import warnings
import ray

ray.init(ignore_reinit_error=True)


class RegularizedMeanEncoder:

    def __init__(self):
        self.levelDict: Dict = {}
        self.defaultPrior: float = None

    @ray.remote
    def _fit_column(self, df, targetLabel, column, a, defaultPrior):
        level_dict = {}
        for level in df[column].unique().tolist():
            if pd.isna(level):
                warnings.warn(
                    f"There are missing values in {column}. Consider converting this to its own level."
                )
            level_dict[level] = defaultPrior

        for category in level_dict.keys():
            level_dict[category] = (df.loc[df[column] == category, targetLabel].sum() + (a * defaultPrior)) / \
                                    (df.loc[df[column] == category].shape[0] + a)
        return column, level_dict

    def fit(self, df, targetLabel, colsToTransform, a=1, defaultPrior=None):

        if defaultPrior is None:
            self.defaultPrior = df[targetLabel].mean()
        else:
            self.defaultPrior = defaultPrior

        tasks = [self._fit_column.remote(df, targetLabel, col, a, self.defaultPrior) for col in colsToTransform]
        results = ray.get(tasks)

        for col, level_dict in results:
            self.levelDict[col] = level_dict

    def transform(self, transformFrame, colsToTransform):
        returnFrame = pd.DataFrame(index=transformFrame.index)

        for col in colsToTransform:
            returnFrame[col + "_enc"] = transformFrame[col].map(self.levelDict[col]).fillna(self.defaultPrior)

        return returnFrame

# Example of usage
if __name__ == "__main__":
    # Creating a mock DataFrame
    df = pd.DataFrame({
        'col1': ['a', 'b', 'a', 'c', 'a', 'b'],
        'col2': ['x', 'y', 'y', 'x', 'x', 'z'],
        'target': [1, 0, 1, 0, 1, 0]
    })

    cols_to_transform = ['col1', 'col2']

    # Initialize the encoder
    encoder = RegularizedMeanEncoder()

    # Fit the encoder
    encoder.fit(df, 'target', cols_to_transform, defaultPrior=1)

    # Transform a DataFrame
    transformed_df = encoder.transform(df, cols_to_transform)
    print(transformed_df)