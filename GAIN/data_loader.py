'''Data loader for UCI letter, spam and MNIST datasets.
'''

########################################################################################################################
# Note: This script is almost a verbatim copy of the counterpart one at https://github.com/jsyoon0823/GAIN,
#       the minimal changes were introduced due to the fact that in our study we used more data per each dataset
#       than what was used in
#       Jinsung Yoon, James Jordon, Mihaela van der Schaar,
#       "GAIN: Missing Data Imputation using Generative Adversarial Nets,"
#       International Conference on Machine Learning (ICML), 2018.
#       The rationale to keep the code, as much as possible, identical to that of (the original) GAIN is to allow
#       fair comparisons.
########################################################################################################################

# Necessary packages
import numpy as np

try:
    from GAIN.utils import binary_sampler
except ModuleNotFoundError:
    from utils import binary_sampler

from typing import Dict, List


DATASETS: Dict[str, List[int]] = {
    "breast": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    # "cover-type": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    #                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    #                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    "credit": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "eeg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "iris": [0, 1, 2, 3],
    "letter": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    # "mushroom": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    #              25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    #              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    #              71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
    #              94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    "news": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
    "spam": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
    "wine-red": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "wine-white": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "yeast": [0, 1, 2, 3, 4, 5, 6, 7]
}


########################################################################################################################
# ORIGINAL IMPLEMENTATION
# def data_loader (data_name, miss_rate):
#     '''Loads datasets and introduce missingness.
#
#     Args:
#       - data_name: letter, spam, or mnist
#       - miss_rate: the probability of missing components
#
#     Returns:
#       data_x: original data
#       miss_data_x: data with missing values
#       data_m: indicator matrix for missing components
#     '''
#
#     # Load data
#     if data_name in ['letter', 'spam']:
#         file_name = 'data/'+data_name+'.csv'
#         data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
#     elif data_name == 'mnist':
#         (data_x, _), _ = mnist.load_data()
#         data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)
#
#     # Parameters
#     no, dim = data_x.shape
#
#     # Introduce missing data
#     data_m = binary_sampler(1-miss_rate, no, dim)
#     miss_data_x = data_x.copy()
#     miss_data_x[data_m == 0] = np.nan
#
#     return data_x, miss_data_x, data_m
########################################################################################################################


########################################################################################################################
# HACK TO ORIGINAL IMPLEMENTATION TO MAKE IT WORK WITH THE SELECTED DATASETS
def data_loader(data_name, miss_rate):
    '''Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    '''

    # Load data
    if data_name in DATASETS.keys():
        file_name = f"datasets/{data_name}.csv"
        try:
            data_x = np.loadtxt(file_name, delimiter=",", skiprows=1, usecols=DATASETS[data_name])
        except OSError:
            data_x = np.loadtxt(f"../{file_name}", delimiter=",", skiprows=1, usecols=DATASETS[data_name])
    else:
        raise ValueError(f"Unsupported dataset, got '{data_name}' and expected one of {list(DATASETS.keys())}.")

    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m

