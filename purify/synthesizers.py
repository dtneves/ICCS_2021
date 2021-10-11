########################################################################################################################
# Research Centers
# ----------------
# Centro ALGORITMI - School of Engineering – University of Minho
# Braga - Portugal
# http://algoritmi.uminho.pt/
#
# Medical Informatics Group
# BIH - Berlin Institute of Health
# Charité - Universitätsmedizin Berlin
# https://www.bihealth.org/en/research/research-groups/fabian-prasser/
#
#
# Description
# -----------
# This module allows to generate synthetic data, more details are provided in [1].
# For the sake of simplicity and for the moment,
# the missingness follows the MCAR -- Missing Completely at Random -- pattern.
# Therefore, each missing value does not depend on the data values.
# The missing rate can range from 0% (0.00) to 99% (0.99).
# On the one hand, it is highly inadvisable to use high values for the missing rate
# since high rates of missingness will lead to a severe loss on data utility.
# On the other hand, very low rates of missingness will likely lead to synthetic data that is highly identifiable.
# Thus, it is up to the user to use metrics that allow him/her to assess
# the trade-off between data utility and data privacy.
# The implementation of advanced techniques to generate synthetic data and,
# yet, maintain as much as possible data privacy is on our research roadmap,
# those will be based on metadata and domain knowledge, amongst other considerations.
# Finally, one should be aware that exception handling to take care of incorrect data types,
# incorrect parameters' values, and so forth is, typically, NOT performed, the rule is:
# We are all grown up (Python) programmers!
#
#
# Moto
# ----
# "We think too much and feel too little. More than machinery we need humanity."
#                         -- Excerpt of the final speech from The Great Dictator
#
#
# Related Work
# ------------
#   * https://link.springer.com/chapter/10.1007/978-3-030-77961-0_10  -->  SGAIN + WSGAIN-CP + WSGAIN-GP paper
#   * https://github.com/dtneves/ICCS_2021
#   * https://arxiv.org/pdf/1907.00503  -->  CTGAN paper
#   * https://github.com/sdv-dev/CTGAN
#   * https://github.com/jsyoon0823/GAIN
#
#
# References
# ----------
#  [1] Diogo Telmo Neves, João Alves, Marcel Ganesh Naik, Alberto Proença, Fabian Praßer.
#      "TODO: TITLE GOES HERE"
#      Journal of Computational Science (JCS), 2022.
#  [2] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença.
#      "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation."
#      International Conference on Computational Science (ICCS), 2021.
#  [3] Jinsung Yoon, James Jordon, Mihaela van der Schaar,
#      "GAIN: Missing Data Imputation using Generative Adversarial Nets."
#      International Conference on Machine Learning (ICML), 2018.
#  [4] Xu, Lei, et al.
#      "Modeling Tabular data using Conditional GAN."
#      Advances in Neural Information Processing Systems (NIPS), 2019.
#
#
# Authors
# -------
# diogo telmo neves -- {dneves@di.uminho.de, diogo-telmo.neves@charite.de}
#
#
# Copyright
# ---------
# Copyright (c) 2020 diogo telmo neves.
# All rights reserved.
#
#
# Conditions
# ----------
# This code is free/open source code but the following conditions must be met:
#   * Redistributions of source code must retain the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#
# DISCLAIMER
# ----------
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
# Date
# ----
# September 2021
########################################################################################################################

import numpy as np

import random

from GAIN.utils import binary_sampler, rounding

from purify.imputation.gain import SGAIN, WSGAIN_CP, WSGAIN_GP

from typing import Any, Callable, Dict, List, Tuple


class SynSGAIN(SGAIN):
    """"This class implements a slightly modified version of the Slim GAIN (SGAIN) algorithm [1, 2],
    which allows to generate synthetic data (i.e., data samples).

    References
    ----------
    [1] Diogo Telmo Neves, João Alves, Marcel Ganesh Naik, Alberto Proença, Fabian Praßer.
        "TODO: TITLE GOES HERE"
        Journal of Computational Science (JCS), 2022.
    [2] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença.
        "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation."
        International Conference on Computational Science (ICCS), 2021.
    """
    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any] = {}):
        super().__init__(data=data, algo_parameters=algo_parameters)
        # # this is NOT needed but it is here since future versions could use it
        # self.n_samples: int = algo_parameters['n_samples'] if 'n_samples' in algo_parameters else 100

    # @classmethod
    # def _execute(cls, data: np.ndarray, algo_parameters: Dict[str, Any] = {}) -> np.ndarray:
    #     data_miss: np.ndarray = data.copy()  # to NOT mess up with the original data
    #     # if NOT given then 20% of missing values will be introduced per each column
    #     miss_rate: float = algo_parameters['miss_rate'] if 'miss_rate' in algo_parameters else 0.2
    #     verbose: bool = algo_parameters['verbose'] if 'verbose' in algo_parameters else False
    #     n_obs: int = data_miss.shape[0]
    #     dim: int = data_miss.shape[1]
    #     indices: Dict[int, Tuple[int, int]] = {}
    #     row: int
    #     col: int
    #     key: int = 0
    #
    #     for row in range(n_obs):
    #         for col in range(dim):
    #             indices[key] = (row, col)
    #             key += 1
    #     if verbose:
    #         print()
    #         print("purify.synthesizers.SynSGAIN :: _execute()")
    #     # at this moment, `data_miss` is only composed by numeric data (i.e., each variable is either of integer or
    #     # float data type), however, since missing values will be introduced, there will be a mismatch when handling
    #     # integer variables, thus all values are casted to `float`
    #     data_miss = data_miss.astype(dtype=float)
    #     while indices:
    #         indices_sample: List[int] = random.sample(
    #             population=list(indices.keys()), k=min(int(n_obs * dim * miss_rate), len(indices)))
    #
    #         # remove each index in `indices_sample` from `indices` and ampute the `data` (i.e., the numpy ndarray)
    #         for index in indices_sample:
    #             row, col = indices.pop(index)   # remove the index
    #             data_miss[row, col] = np.NaN    # ampute the cell mapped by the index
    #         if verbose:
    #             print()
    #             print(data_miss[0:5, :])
    #             print("...")
    #             print(data_miss.shape)
    #         data_miss = SynSGAIN(data=data_miss, algo_parameters=algo_parameters).execute()
    #         if verbose:
    #             print()
    #             print(data_miss[0:5, :])
    #             print("...")
    #             print(data_miss.shape)
    #     return data_miss

    # @classmethod
    # def sampler(cls, data: np.ndarray, algo_parameters: Dict[str, Any] = {}) -> np.ndarray:
    #     # if NOT given then 100 samples will be generated
    #     n_samples: int = algo_parameters['n_samples'] if 'n_samples' in algo_parameters else 100
    #     verbose: bool = algo_parameters['verbose'] if 'verbose' in algo_parameters else False
    #     n_obs: int = data.shape[0]
    #     remainder: int = n_samples % n_obs
    #     synthetic_data: np.ndarray
    #
    #     synthetic_data_list: List[np.ndarray] = [
    #         SynSGAIN._execute(data=data, algo_parameters=algo_parameters) for _ in range(n_samples // n_obs)]
    #     if remainder > 0:
    #         random_indices: np.ndarray = np.random.choice(a=n_obs, size=remainder, replace=False)
    #
    #         synthetic_data_list.append(
    #             SynSGAIN._execute(data=data, algo_parameters=algo_parameters)[random_indices, :])
    #     synthetic_data = np.concatenate(synthetic_data_list, axis=0)
    #     if verbose:
    #         print()
    #         print("purify.synthesizers.SynSGAIN :: sampler()")
    #         print(synthetic_data[0:5, :])
    #         print("...")
    #         print(synthetic_data.shape)
    #     return synthetic_data


class SynSGAIN_CP(WSGAIN_CP):

    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any] = {}):
        super().__init__(data=data, algo_parameters=algo_parameters)
        # # this is NOT needed but it is here since future versions could use it
        # self.n_samples: int = algo_parameters['n_samples'] if 'n_samples' in algo_parameters else 100


class SynSGAIN_GP(WSGAIN_GP):

    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any] = {}):
        super().__init__(data=data, algo_parameters=algo_parameters)
        # # this is NOT needed but it is here since future versions could use it
        # self.n_samples: int = algo_parameters['n_samples'] if 'n_samples' in algo_parameters else 100


class Synthesizer:

    SYNTHESIZERS: Dict[str, Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = {
        'SynSGAIN': SynSGAIN,
        'SynSGAIN-CP': SynSGAIN_CP,
        'SynSGAIN-GP': SynSGAIN_GP
    }
    """The supported data synthesizers (i.e., data generators)."""

    def __init__(self, data: np.ndarray, algo: str = 'SynSGAIN', algo_parameters: Dict[str, Any] = {}):
        self.data: np.ndarray = data.copy()  # to NOT mess up with the given `data`
        # if algo not in Synthesizer.SYNTHESIZERS.keys():
        #     raise ValueError("Expecting one of the supported data synthesizers -- "
        #                      f"{' ,'.join([Synthesizer.SYNTHESIZERS.keys()])} -- "
        #                      f"as the algorithm to be used for data generation but got: {algo}.")
        self.algo: Callable[[np.ndarray, Dict[str, Any]], np.ndarray] = Synthesizer.SYNTHESIZERS[algo] \
            if algo in Synthesizer.SYNTHESIZERS.keys() else SynSGAIN
        self.algo_parameters: Dict[str, Any] = algo_parameters
        ################################################################################################################
        # if NOT given then 20% of missing values will be introduced per each column
        # self.miss_rate: float = algo_parameters['miss_rate'] if 'miss_rate' in algo_parameters else 0.2
        # if NOT given then 100 samples will be generated
        self.n_samples: int = algo_parameters['n_samples'] if 'n_samples' in algo_parameters else 100
        ################################################################################################################
        # TODO: VERIFY IF THIS CAN BE REMOVED --> LOOK AT THE `SGAIN` IMPLEMENTATION
        ################################################################################################################
        self.verbose: bool = algo_parameters['verbose'] if 'verbose' in algo_parameters else False
        self.n_obs: int = self.data.shape[0]
        self.dim: int = self.data.shape[1]
        self.remainder: int = self.algo_parameters['n_samples'] % self.n_obs

    def _execute(self) -> np.ndarray:
        synthetic_data: np.ndarray = self.data.copy()
        indices: Dict[int, Tuple[int, int]] = {}
        row: int
        col: int
        key: int = 0

        for row in range(self.n_obs):
            for col in range(self.dim):
                indices[key] = (row, col)
                key += 1
        if self.verbose:
            print()
            print("purify.synthesizers.Synthesizer :: _execute()")
        while indices:
            indices_sample: List[int] = random.sample(
                population=list(indices.keys()),
                k=min(int(self.n_obs * self.dim * self.algo_parameters['miss_rate']), len(indices)))
            # for each run there is the need of using a fresh copy of the original data (i.e., the synthetic data
            # will be always drawn from the original data)
            # additionally, after the preprocessing stage, the original data is only composed by numeric data
            # (i.e., each variable is either of `int` or of `float` data type) yet there is the need to ensure
            # that it is only of `float` data type, otherwise there will be a data type mismatch
            # when introducing missing values into an `int` variable
            data: np.ndarray = self.data.copy().astype(dtype=float)
            positions: List[int, int] = []

            # remove each index in `indices_sample` from `indices` and ampute the `data` (i.e., the numpy ndarray)
            for index in indices_sample:
                row, col = indices.pop(index)   # remove the index
                data[row, col] = np.NaN         # ampute the cell mapped by the index
                positions.append((row, col))
            if self.verbose:
                print()
                print(f"first {min(5, self.algo_parameters['n_samples'])} row(s) of synthetic data:")
                print(synthetic_data[0:min(5, self.algo_parameters['n_samples']), :])
                print("...")
                print(f"shape: {synthetic_data.shape}")
            data = self.algo(data=data, algo_parameters=self.algo_parameters).execute()
            for row, col in positions:
                synthetic_data[row, col] = data[row, col]
            if self.verbose:
                print()
                print(f"first {min(5, self.algo_parameters['n_samples'])} row(s) of synthetic data:")
                print(synthetic_data[0:min(5, self.algo_parameters['n_samples']), :])
                print("...")
                print(f"shape: {synthetic_data.shape}")
        return synthetic_data

    def sampler(self) -> np.ndarray:
        synthetic_data_list: List[np.ndarray] = [
            self._execute() for _ in range(self.algo_parameters['n_samples'] // self.n_obs)]
        synthetic_data: np.ndarray

        if self.remainder > 0:
            random_indices: np.ndarray = np.random.choice(a=self.n_obs, size=self.remainder, replace=False)

            # slice from the output of this final run using random indices of it
            synthetic_data_list.append(self._execute()[random_indices, :])
        synthetic_data = np.concatenate(synthetic_data_list, axis=0)
        if self.verbose:
            print()
            print("purify.synthesizers.Synthesizer :: sampler()")
            print(f"algo:  {self.algo}")
            print(f"first {min(5, self.n_samples)} row(s) of synthetic data:")
            print(synthetic_data[0:min(5, self.n_samples), :])
            print("...")
            print(f"shape: {synthetic_data.shape}")
        return synthetic_data


