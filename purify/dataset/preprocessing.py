
import numpy as np
import pandas as pd

from purify.dataset.metadata import Metadata


class PreProcessor:
    """This class provides a few methods that leverage a few data encoders operations.
    One should be aware that the implementation of this class is dependent (i.e., it relies) on
    the implementation of the :class:`purify.dataset.metadata.Metadata` class.
    """

    @classmethod
    def drop_columns(cls, df: pd.DataFrame, dataset: str, verbose: bool = False) -> pd.DataFrame:
        """Drop each column (i.e., each feature/variable) that is marked in its metadata to be dropped.
        One should be aware that this implementation is only for datasets that are supported through
        the :class:`purify.dataset.metadata.Metadata` class.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame with data of the given `dataset`.
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported through
            the :class:`purify.dataset.metadata.Metadata` class.
        verbose : bool, optional
            If True some info will be sent to the standard output, which is useful, for instance, to debug and
            to trace the execution.

        Returns
        -------
        df_drop : DataFrame
            A copy of the given pandas DataFrame (i.e., of the given `df`) in which each observation (i.e., each row)
            that has at least one missing value (i.e., a numpy NaN).

        See Also
        --------
        :func:`purify.dataset.metadata.Metadata.vars_to_drop`
        """
        df_drop: pd.DataFrame = df.drop(columns=Metadata.vars_to_drop(dataset=dataset, df=df))

        if verbose:
            print()
            print("purify.dataset.encoders.PreProcessor :: drop_columns()")
            print("Before dropping columns:")
            print(df.head())
            print("...")
            print(df.tail())
            print("After dropping columns:")
            print(df_drop.head())
            print("...")
            print(df_drop.tail())
        return df_drop

    @classmethod
    def drop_nans(cls, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Drop each observation (i.e., each row) that has at least one missing value (i.e., a numpy NaN).
        One should be aware that this implementation is only for datasets that are supported through
        the :class:`purify.dataset.metadata.Metadata` class.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output, which is useful, for instance, to debug and
            to trace the execution.

        Returns
        -------
        df_drop : DataFrame
            A copy of the given pandas DataFrame (i.e., of the given `df`) in which each observation (i.e., each row)
            that has at least one missing value (i.e., a numpy NaN) is dropped.
        """
        df_drop: pd.DataFrame = df.dropna()

        if verbose:
            print()
            print("purify.dataset.encoders.PreProcessor :: drop_nans()")
            print("Before dropping NaNs:")
            print(df.head())
            print("...")
            print(df.tail())
            print("After dropping NaNs:")
            print(df_drop.head())
            print("...")
            print(df_drop.tail())
        return df_drop

    @classmethod
    def replace_miss_values_by_nans(cls, df: pd.DataFrame, dataset: str, verbose: bool = False) -> pd.DataFrame:
        """Replace the missing values of each column (i.e., of each feature/variable) by numpy NaNs.
        One should be aware that this implementation is only for datasets that are supported through
        the :class:`purify.dataset.metadata.Metadata` class.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame with data of the given `dataset`.
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported through
            the :class:`purify.dataset.metadata.Metadata` class.
        verbose : bool, optional
            If True some info will be sent to the standard output, which is useful, for instance, to debug and
            to trace the execution.

        Returns
        -------
        df_rep : DataFrame
            A copy of the given pandas DataFrame (i.e., of the given `df`) in which
            the missing values of each column (i.e., of each feature/variable) are replaced by numpy NaNs.
        """
        df_rep: pd.DataFrame = df.copy(deep=True)

        for variable in Metadata.DATASETS[dataset].keys():
            # it could happen that, for instance, the :func:`purify.encoders.PreProcessor.vars_to_drop`
            # method has been invoked before this one and, if so, the current `variable` may NOT be part of
            # the given pandas DataFrame (i.e., may NOT be a column of `df`),
            # thus, it is required to perform the following test
            if variable in df_rep.columns:
                for value in Metadata.DATASETS[dataset][variable]['missing_values']:
                    df_rep[variable] = df_rep[variable].replace(to_replace=value, value=np.NaN)
        if verbose:
            print()
            print("purify.dataset.encoders.PreProcessor :: replace_miss_values_by_nans()")
            print("Before replacing missing values:")
            print(df.head())
            print("...")
            print(df.tail())
            print("After replacing missing values:")
            print(df_rep.head())
            print("...")
            print(df_rep.tail())
        return df_rep

