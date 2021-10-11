########################################################################################################################
# Research Centers:
# -----------------
# Centro ALGORITMI - School of Engineering – University of Minho
# Braga - Portugal
# http://algoritmi.uminho.pt/
#
# Medical Informatics Group
# BIH - Berlin Institute of Health
# Charité - Universitätsmedizin Berlin
# https://www.bihealth.org/en/research/research-groups/fabian-prasser/
#
# Description:
# ------------
# This module allows to perform basic data encoders operations -- label encoding and one-hot encoding -- and
# to inverse (i.e., to revert) those data transformations.
# One should be aware that exception handling to take care of incorrect data types, incorrect parameters' values, and
# so forth is, typically, NOT performed, the rule is: We are all grown up (Python) programmers!
#
#
# Moto:
# -----
# "We think too much and feel too little. More than machinery we need humanity."
#                         -- Excerpt of the final speech from The Great Dictator
#
#
# Related Work:
# -------------
#   * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
#   * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#   * https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
#
#
# References:
# -----------
#  [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
#      "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
#      International Conference on Computational Science (ICCS), 2021.
#
#
# Authors:
# --------
# diogo telmo neves -- {dneves@di.uminho.de, diogo-telmo.neves@charite.de}
#
#
# Copyright:
# ----------
# Copyright (c) 2021 diogo telmo neves.
# All rights reserved.
#
#
# Conditions:
# -----------
# This code is free/open source code but the following conditions must be met:
#   * Redistributions of source code must retain the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#
# DISCLAIMER:
# -----------
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Date:
# -----
# September 2021
########################################################################################################################

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from typing import Any, Dict, List, Tuple, Union


def label_encoders_fit_transform(data: pd.DataFrame,
                                 discrete_cols: Union[List[str], List[int]],
                                 verbose: bool = False) -> Tuple[pd.DataFrame, Dict[Union[str, int], LabelEncoder]]:
    """Applies label encoding to each of the given `discrete_cols` of the given `data`,
    following the paradigm 'fit and transform'.

    :param data: The data (as a `pd.DataFrame`) to be transformed.
    :param discrete_cols: The discrete columns (aka the categorical features/variables) of the given `data`.
    It is important to notice that the list is either a list of columns' names (i.e., a list of `str`) or
    a list of columns indices (i.e., a list of `int`).
    :param verbose: To control the verbosity -- amongst other usages, it is useful to debug.
    :return: A tuple that is composed by the result of the data transformation as well as
    by a dictionary that maps each column's name or column's index to an instance of `LabelEncoder`,
    which allows to invert (i.e., to revert) the transformation.
    The dictionary can be depicted as follows: {<column's name> | <column's index>: <label encoder>, ...}
    """
    # more than just sanity checks
    if isinstance(data, pd.DataFrame):  # pd.DataFrame
        df: pd.DataFrame = data.copy(deep=True)  # to NOT mess up the given data structure
    else:  # NOT an pd.DataFrame
        raise ValueError(f"Expecting a pandas DataFrame but got: {type(data)}.")
    if set(discrete_cols) - set(list(df.columns)):
        raise ValueError("Bad list of discrete columns, "
                         "at least one of them does NOT belong to the columns of the given `data`.")

    label_encoders: Dict[Union[str, int], LabelEncoder] = {}

    if verbose:
        print("\nBefore applying label encoder:")
        print(df.head())
        print("...")
        print(df.tail())
    for discrete_col in discrete_cols:
        label_encoders[discrete_col] = LabelEncoder()
        df[discrete_col] = label_encoders[discrete_col].fit_transform(y=df[discrete_col])
    if verbose:
        print("\nAfter applying label encoder:")
        print(df.head())
        print("...")
        print(df.tail())
    return df, label_encoders


def label_encoders_inverse_transform(data: pd.DataFrame,
                                     label_encoders: Dict[Union[str, int], LabelEncoder],
                                     verbose: bool = False) -> pd.DataFrame:
    """Applies an inverse transformation to the given `data` using the given `label_encoders`.

    :param data: The data (as a `pd.DataFrame`) to be transformed.
    :param label_encoders: A dictionary that maps each column's name or column's index to an instance of `LabelEncoder`,
    which allows to invert (i.e., to revert) the (previously applied data) transformation.
    The dictionary can be depicted as follows: {<column's name> | <column's index>: <label encoder>, ...}
    :param verbose: To control the verbosity -- amongst other usages, it is useful to debug.
    :return: An instance of `pd.DataFrame` with the result of the inverse (i.e., the reverse) transformation.
    """
    # more than just sanity checks
    if isinstance(data, pd.DataFrame):  # pd.DataFrame
        df: pd.DataFrame = data.copy(deep=True)  # to NOT mess up the given data structure
    else:  # NOT an pd.DataFrame
        raise ValueError(f"Expecting a pandas DataFrame but got: {type(data)}.")
    if set(label_encoders.keys()) - set(list(df.columns)):
        raise ValueError("At least one of the discrete columns does NOT belong to the columns of the given `data`.")

    if verbose:
        print("\nBefore applying inverse transform:")
        print(df.head())
        print("...")
        print(df.tail())
    for discrete_col, label_encoder in label_encoders.items():
        df[discrete_col] = label_encoder.inverse_transform(y=data[discrete_col])
    if verbose:
        print("\nAfter applying inverse transform:")
        print(df.head())
        print("...")
        print(df.tail())
    return df


def get_dummies_fit_transform(data: pd.DataFrame,
                              discrete_cols: Union[List[str], List[int]],
                              verbose: bool = False) -> pd.DataFrame:
    """Applies a data transformation that is identically to the well known One-Hot Encoding data transformation.

    DISCLAIMER: THIS FUNCTION DOES NOT PERFORM A `FIT AND TRANSFORM` OPERATION,
                ITS BEHAVIOR LOOKS LIKE A `FIT AND TRANSFORM` OPERATION BUT IT IS NOT.

    :param data: The data (as a `pd.DataFrame`) to be transformed.
    :param discrete_cols: The discrete columns (aka the categorical features/variables) of the given `data`,
    expressed either as a list of columns' names (i.e., a list of `str`) or
    as a list of columns' indices (i.e., a list of `int`).
    :param verbose: To control the verbosity -- amongst other usages, it is useful to debug.
    :return: An instance of `pd.DataFrame` with the result of the (alike) 'fit and transform' data transformation.
    """
    # just a sanity check
    if not isinstance(data, pd.DataFrame):  # NOT an pd.DataFrame
        raise ValueError(f"Expecting a pandas DataFrame but got: {type(data)}.")

    df: pd.DataFrame = pd.get_dummies(data=data, columns=discrete_cols)

    if verbose:
        print("\nBefore applying fit transform:")
        print(data.head())
        print("...")
        print(data.tail())
        print("\nAfter applying fit transform:")
        print(df.head())
        print("...")
        print(df.tail())
    return df


def get_dummies_inverse_transform(data: pd.DataFrame,
                                  discrete_cols: List[Tuple[Union[str, int], Any]],
                                  columns_order: Union[List[str], List[int]] = None,
                                  verbose: bool = False) -> pd.DataFrame:
    """Applies a data transformation that is identically to the well known inverse transformation that is possible to
    apply after applying the One-Hot Encoding data transformation based on the 'fit and transform' paradigm.

    DISCLAIMER: THIS FUNCTION DOES NOT PERFORM TRULY AN `INVERSE TRANSFORM` OPERATION,
                HOWEVER ITS BEHAVIOR LOOKS LIKE INDEED AN `INVERSE TRANSFORM` OPERATION.

    :param data: The data (as a `pd.DataFrame`) to be transformed.
    :param discrete_cols: The discrete columns (aka the categorical features/variables) of the given `data`.
    It is important to notice that each element of the list is a tuple that is composed by
    a column's name (as a `str`) or a column's index (as an `int`) and the column's data type,
    the data structure can be depicted as follows: [(<column's name> | <column's index>, <column's data type>), ...]
    :param columns_order: The desired columns' order of the result (i.e., pandas DataFrame).
    :param verbose: To control the verbosity -- amongst other usages, it is useful to debug.
    :return: An instance of `pd.DataFrame` with the result of the (alike) inverse (i.e., the reverse) transformation.
    """
    # just a sanity check
    if not isinstance(data, pd.DataFrame):  # NOT an pd.DataFrame
        raise ValueError(f"Expecting a pandas DataFrame but got: {type(data)}.")

    df: pd.DataFrame = data.copy(deep=True)  # to NOT mess up the given data structure
    df_final: pd.DataFrame = pd.DataFrame()

    if verbose:
        print("\nBefore applying inverse transform:")
        print(df.head())
        print("...")
        print(df.tail())
    for col, data_type in discrete_cols:
        # finds the columns associated with the current `col`
        columns: List[str] = [column for column in df.columns if column.startswith(f"{col}_")]
        # creates a pandas DataFrame with the columns from which is possible to invert the transformation
        df_tmp: pd.DataFrame = pd.DataFrame(data=df[columns])
        # most important statement of this algorithm,
        # the first part -- `df_tmp.idxmax(axis=1)` -- does the magic of inverting the transformation,
        # whereas the second part removes the unwanted string from the head of the string of each cell
        df_tmp[col] = df_tmp.idxmax(axis=1).map(lambda x: x.replace(f"{col}_", ""))
        # drops the unwanted columns and concatenates both pandas DataFrames
        df_final = pd.concat(objs=[df_final, df_tmp.drop(columns=columns)], axis=1)
        # ensure that the discrete column is of type `int`
        df_final[col] = df_final[col].astype(dtype=data_type)
    # adds also the continuous columns and, then, put them all in the initial columns' order
    df_final = pd.concat(objs=[df[set(columns_order) - set([column for column, data_type in discrete_cols])], df_final],
                         axis=1)[columns_order]
    if verbose:
        print("\nAfter applying inverse transform:")
        print(df_final.head())
        print("...")
        print(df_final.tail())
    return df_final


def __test_minimal_example() -> None:
    df: pd.DataFrame = pd.DataFrame(data=['Male', 'Female', 'Female', 'Female', 'Male', 'Female'], columns=['Gender'])
    columns_order: Union[List[str], List[int]] = df.columns
    # [(<column's name> | <column's index>, <data type>), ...]
    discrete_columns: List[Tuple[Union[str, int], Any]] = [('Gender', str)]

    print(pd.get_dummies(data=df))
    df = get_dummies_fit_transform(data=df, discrete_cols=[column for column, data_type in discrete_columns])
    print(df)
    df = get_dummies_inverse_transform(data=df, discrete_cols=discrete_columns, columns_order=columns_order)
    print(df)


def __test_adult_example_basic() -> None:
    # data = load_demo()  # adult dataset from CTGAN
    data: pd.DataFrame = pd.read_csv("./datasets/adult.csv")  # adult dataset from UCI
    # original columns' order
    columns_order: List[str] = data.columns
    # a list of each column that is discrete and its data type
    discrete_columns: List[Tuple[str, Any]] = [
        ('workclass', str),
        ('education', str),
        ('marital-status', str),
        ('occupation', str),
        ('relationship', str),
        ('race', str),
        ('sex', str),
        ('native-country', str),
        ('income', str)
    ]

    print()
    print("--- FIRST STAGE ---" * 3)

    first_stage_out: pd.DataFrame = get_dummies_fit_transform(
        data=data, discrete_cols=[column for column, data_type in discrete_columns], verbose=True)

    print()
    print("--- SECOND STAGE ---" * 3)

    second_stage_out: pd.DataFrame = get_dummies_inverse_transform(
        data=first_stage_out, discrete_cols=discrete_columns, columns_order=columns_order, verbose=True)


def __test_adult_example_advanced() -> None:
    # data = load_demo()  # adult dataset from CTGAN
    data: pd.DataFrame = pd.read_csv("./datasets/adult.csv")  # adult dataset from UCI
    # original columns' order
    columns_order: List[str] = data.columns
    # a list of each column that is discrete and its data type
    # one should notice that, when comparing this example against the 'basic' one,
    # the data type is `int` and NOT `str` because the data type needs to be
    # according to the previous data transformation applied to the data that flows within the pipeline
    discrete_columns: List[Tuple[str, Any]] = [
        ('workclass', int),
        ('education', int),
        ('marital-status', int),
        ('occupation', int),
        ('relationship', int),
        ('race', int),
        ('sex', int),
        ('native-country', int),
        ('income', int)
    ]

    print()
    print("--- FIRST STAGE ---" * 3)

    first_stage_out: Tuple[pd.DataFrame, Dict[Union[str, int], LabelEncoder]] = label_encoders_fit_transform(
        data=data, discrete_cols=[column for column, data_type in discrete_columns], verbose=True)

    print()
    print("--- SECOND STAGE ---" * 3)

    second_stage_out: pd.DataFrame = get_dummies_fit_transform(
        data=first_stage_out[0], discrete_cols=[column for column, data_type in discrete_columns], verbose=True)

    print()
    print("--- THIRD STAGE ---" * 3)

    third_stage_out: pd.DataFrame = get_dummies_inverse_transform(
        data=second_stage_out, discrete_cols=discrete_columns, columns_order=columns_order, verbose=True)

    print()
    print("--- FOURTH STAGE ---" * 3)

    fourth_stage_out = label_encoders_inverse_transform(
        data=third_stage_out, label_encoders=first_stage_out[1], verbose=True)

