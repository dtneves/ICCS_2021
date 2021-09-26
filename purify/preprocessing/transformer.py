import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils._testing import ignore_warnings


class DataTransformer:
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.

    Args:
        n_cluster (int):
            Number of modes.
        epsilon (float):
            Epsilon value.
    """

    @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, data: np.ndarray):
        scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1.00, +1.00))
        scaler.fit(X=data)

        return {
            'model': scaler,
            'output_info': [(1, 'tanh')],
            'output_dimensions': 1
        }

    def _fit_discrete(self, data: np.ndarray):
        categories: np.ndarray = [list(filter(lambda x: not np.isnan(x), np.unique(data)))]
        ohe: OneHotEncoder = OneHotEncoder(categories=categories, sparse=False, handle_unknown='ignore')
        ohe.fit(X=data)
        categories = len(ohe.categories_[0])

        return {
            'encoder': ohe,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories
        }

    def fit(self, data: np.ndarray, discrete_columns):
        if not discrete_columns: discrete_columns = []
        self.output_info = []
        self.output_dimensions = 0

        for idx, column_data in enumerate(data.T):
            column: np.ndarray = column_data.reshape(-1, 1)
            if idx in discrete_columns:
                meta = self._fit_discrete(column)
            else:
                meta = self._fit_continuous(column)

            self.output_info.append(meta)
            self.output_dimensions += meta['output_dimensions']

        return self

    def _transform_continuous(self, column_meta: dict, data: np.ndarray):
        model: MinMaxScaler = column_meta['model']
        return model.transform(data)

    def _transform_discrete(self, column_meta: dict, data: np.ndarray):
        encoder: OneHotEncoder = column_meta['encoder']
        t = encoder.transform(data)
        t[np.where(~t.any(axis=1))[0]] = np.nan
        return t

    def transform(self, data):
        values = []
        for idx, info in enumerate(self.output_info):
            column_data = data[:, idx]
            column = column_data.reshape(-1, 1)
            if 'model' in info:
                values.append(self._transform_continuous(info, column))
            else:
                values.append(self._transform_discrete(info, column))

        return np.concatenate(values, axis=1).astype(float)

    def fit_transform(self, data, discrete_columns):
        return self.fit(data, discrete_columns=discrete_columns).transform(data)

    def _inverse_transform_continuous(self, info, data):
        model = info['model']
        return model.inverse_transform(data)

    def _inverse_transform_discrete(self, info, data):
        encoder = info['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(self, data):
        start = 0
        output = []
        for info in self.output_info:
            dimensions = info['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in info:
                inverted = self._inverse_transform_continuous(info, columns_data)
            else:
                inverted = self._inverse_transform_discrete(info, columns_data)

            n_unique = np.unique(inverted)
            output.append(inverted)
            start += dimensions

        return np.column_stack(output)
