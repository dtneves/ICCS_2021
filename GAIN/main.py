"""Main function for UCI letter and spam datasets.
"""

########################################################################################################################
# Note: This script is almost a verbatim copy of the counterpart one at https://github.com/jsyoon0823/GAIN,
#       the minimal changes wer introduced to make the code work and to enable more than one run.
#       The noteworthy ones are all concerning multiple runs, see below lines 49, 58, 69, and 129-133.
#       Multiple runs imply multiple imputations and, therefore, multiple RMSE.
#       Thus, there are variables to hold those multiple results, see below lines: 66-69, 79-80, 82 and 92.
#       The rationale to keep the code, as much as possible, identical to that of (the original) GAIN is to allow
#       fair comparisons.
########################################################################################################################

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np

from tqdm import tqdm

from data_loader import data_loader
from gain import gain
from utils import rmse_loss

from time import time

from typing import Any, Dict, List

########################################################################################################################
# Note: This is a dumb hack to enable running the code in my old Mac Air without having to do a lot of reconfiguration.
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
########################################################################################################################


def main(args):
    """Main function not only for UCI letter and spam datasets.

    Args:
      - data_name: the short name of a dataset
      - miss_rate: probability of missing components
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyper-parameter
      - iterations: iterations
      - n_runs: n runs

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    """
    data_name: str = args.data_name
    miss_rate: float = args.miss_rate
    n_runs: int = args.n_runs
    gain_parameters: Dict[str, Any] = {
        'batch_size': args.batch_size,
        'hint_rate': args.hint_rate,
        'alpha': args.alpha,
        'n_iterations': args.n_iterations}
    imp_data_lst: List[np.ndarray] = []
    imp_data_lst_append = imp_data_lst.append
    rmse_lst: List[float] = []
    rmse_lst_append = rmse_lst.append
    time_lst: List[float] = []
    time_lst_append = time_lst.append
    t0: float

    for _ in tqdm(range(n_runs)):
        t0 = time()
        # Load data and introduce missingness
        ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
        # Impute missing data
        imputed_data_x = gain(miss_data_x, gain_parameters)
        # Report the RMSE performance
        rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)

        imp_data_lst_append(imputed_data_x)
        rmse_lst_append(rmse)
        time_lst_append(time() - t0)

    imputed_data: np.ndarray = np.sum(imp_data_lst, axis=0) / len(imp_data_lst)

    print("GAIN:")
    print(f"\tdataset:    {args.data_name}")
    print(f"\tshape:      {ori_data_x.shape}")
    print(f"\tmiss rate:  {args.miss_rate:.2f}")
    print(f"\tbatch size: {args.batch_size}")
    print(f"\talpha:      {args.alpha}")
    print(f"\t# iters.:   {args.n_iterations}")
    print(f"\t# runs:     {args.n_runs}")
    print(f"\tRMSE:       {np.mean(rmse_lst):.4f} ({np.std(rmse_lst):.4f})")
    print(f"\tRMSE list:  {rmse_lst}")
    print(f"\ttime (s):   {np.mean(time_lst):.4f} ({np.std(time_lst):.4f})")
    print(f"\ttime list:  {time_lst}")
    return imputed_data, rmse


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_name',
        choices=['breast', 'credit', 'eeg', 'iris', 'letter', 'news', 'spam', 'wine-red', 'wine-white', 'yeast'],
        default='iris',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--n_iterations',
        help='number of training interations',
        default=10000,
        type=int)
    parser.add_argument(
        '--n_runs',
        help='number of runs',
        default=3,
        type=int)

    args = parser.parse_args()

    # Calls main function
    imputed_data, rmse = main(args)


# time python3 main.py --data_name=iris --miss_rate=0.2 --n_iterations=1000 --n_runs=3
