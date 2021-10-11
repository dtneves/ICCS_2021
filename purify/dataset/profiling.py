
import pandas as pd

from typing import Any, Dict, List, Union


def profiler_continuous_variable(series: pd.Series) -> Dict[str, float]:
    """Computes some metadata from the series of values of a continuous variable (i.e., feature/column).

    Parameters
    ----------
    series : pd.Series
        The series of values of a continuous variable (i.e., feature/column).

    Returns
    -------
    Dict[str, float]:
        A dictionary with the minimum, the maximum, the mean, the median, the standard deviation, the skewness, and
        the kurtosis values of the given pandas Series (i.e., variable/feature/column).
    """
    if not str(series.dtypes).startswith("int") and not str(series.dtypes).startswith("float"):
        raise ValueError("Expecting a pandas Series of integers or floating points numbers "
                         f"but got: {series.dtypes}.")
    return {
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "skewness": series.skew(),
        "kurtosis": series.kurtosis()
    }


def profiler_discrete_variable(series: pd.Series) -> Dict[Any, int]:
    """Computes some metadata from the series of values of a discrete variable (i.e., feature/column).

    Parameters
    ----------
    series : pd.Series
        The series of values of a discrete variable (i.e., feature/column).

    Returns
    -------
    Dict[Any, int]:
        A dictionary with the values counts of the given pandas Series (i.e., variable/feature/column).
    """
    return {value: count for value, count in series.value_counts().items()}


def profiler(
        df: pd.DataFrame,
        discrete_vars: Union[List[str], List[int]] = []) -> Union[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Computes some metadata for each variable (i.e., feature/column) of the given pandas DataFrame (i.e., `df`).

    Parameters
    ----------
    df : pd.DataFrame
        The data from which the metadata will be computed.
    discrete_vars : Union[List[str], List[int]]
        The list of discrete variables (i.e., features/columns) of the given pandas DataFrame (i.e., `df`).

    Returns
    -------
    Union[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        A dictionary with some metadata concerning the values distribution of each variable (i.e., feature/column)
        of the given pandas DataFrame (i.e., `df`).
    """
    if not set(discrete_vars).issubset(df.columns):
        raise ValueError("At least one variable of the given set of discrete variables (i.e., features/columns) "
                         "is NOT a variable (i.e., is NOT a feature/column) of the given pandas DataFrame.")
    return {
        variable: profiler_discrete_variable(series=df[variable]) if variable in discrete_vars
        else profiler_continuous_variable(series=df[variable]) for variable in df.columns
    }

