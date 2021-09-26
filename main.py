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
# Intelligent Analytics for Massive Data -- IAM
# German Research Center for Artificial Intelligence -- DFKI
# Deutsches Forschungszentrum für Künstliche Intelligenz -- DFKI
# https://www.dfki.de/web/
#
#
# Description:
# ------------
# This Python script is the program that allows to run the experiments described in [1].
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
#   * https://github.com/epsilon-machine/missingpy
#   * https://github.com/eltonlaw/impyute
#   * https://github.com/iskandr/fancyimpute
#   * https://github.com/kearnz/autoimpute
#   * https://github.com/awslabs/datawig
#   * https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute
#   * https://www.statsmodels.org/stable/api.html?#imputation
#   * https://github.com/jsyoon0823/GAIN
#
#
# References:
# -----------
#  [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
#      "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
#      International Conference on Computational Science (ICCS), 2021.
#  [2] Jinsung Yoon, James Jordon, Mihaela van der Schaar,
#      "GAIN: Missing Data Imputation using Generative Adversarial Nets,"
#      International Conference on Machine Learning (ICML), 2018.
#  [3] Rubin, Donald B. "Inference and missing data." Biometrika 63.3 (1976): 581-592.
#  [4] Van Buuren, Stef. Flexible imputation of missing data. Chapman and Hall/CRC, 2018.
#
#
# Authors:
# --------
# diogo telmo neves -- {dneves@di.uminho.de, diogo-telmo.neves@charite.de}
#
#
# Copyright:
# ----------
# Copyright (c) 2020 diogo telmo neves.
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
# December 2020
########################################################################################################################

from argparse import ArgumentParser, Namespace

import numpy as np

import pandas as pd

from ctgan import CTGANSynthesizer

from GAIN.data_loader import data_loader
from GAIN.gain import gain
from GAIN.utils import rmse_loss

from purify.imputation.gain import SGAIN, WSGAIN_CP, WSGAIN_GP
from purify.preprocessing import DataTransformer

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from tqdm import tqdm

from time import time

from typing import Any, Callable, Dict, List, Set, Tuple, Union

########################################################################################################################
# a few datasets from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/index.php) and their metadata.
# the metadata is useful for some basic data preprocessing tasks (e.g., label, ordinal, or
# one-hot encoding of categorical variables, and min-max normalization of independent variables) as well as
# for model instantiation and configuration.
# the models are according to what was used in https://arxiv.org/abs/1806.02920 and in https://arxiv.org/abs/2006.11783
########################################################################################################################
DATASETS: Dict[str, Dict[str, Any]] = {
    "breast": {
        "name": "Breast Cancer Wisconsin (Diagnostic) Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
        "header": [0],
        "drop_cols": ["ID"],  # columns to drop
        "categorical_vars": {  # the cat. (i.e., discrete) vars. (i.e., features) that need to be encoded
        #     "Diagnosis": {
        #         "class": LabelEncoder,
        #         "kwargs": {},
        #     }
        },
        "target": "Diagnosis",  # the  label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        "model": {
            "class": LogisticRegression,
            "kwargs": {
                "class_weight": None,  # class_weight: dict or ‘balanced’, default=None
                "max_iter": 3000,  # max_iter: int, default=100
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "covertype": {
        "name": "Covertype Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/Covertype",
        "header": [0],
        "drop_cols": [],  # columns to drop
        "categorical_vars": None,  # in this case, for the sake of simplicity we do NOT enumerate the cat. vars.
        "target": "Cover_Type",  # the  label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        "model": {
            "class": LogisticRegression,
            "kwargs": {
                "class_weight": None,  # class_weight: dict or ‘balanced’, default=None
                "max_iter": 1000,  # max_iter: int, default=100
                "n_jobs": -1  # -1 means using all processors
            }
        }
        # "model": {
        #     "class": KNeighborsClassifier,
        #     "kwargs": {
        #         "n_neighbors": 5,
        #         "weights": 'uniform',    # weights: {‘uniform’, ‘distance’} or callable, default=’uniform’
        #         "algorithm": 'auto',     # algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
        #         "p": 2,                  # p: int, default=2 (p = 1 --> l1, manhattan; p = 2 --> l2, euclidean)
        #         "n_jobs": -1             # -1 means using all processors
        #     }
        # }
    },
    "credit": {  # TODO: NEWS MORE ITERATIONS FOR THE MODEL TO CONVERGE
        "name": "Default of Credit Card Clients Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients",
        "header": [0],
        "drop_cols": ["ID"],  # columns to drop
        "categorical_vars": {  # the cat. (i.e., discrete) vars. (i.e., features) that need to be encoded
            "SEX": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "EDUCATION": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "MARRIAGE": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "PAY_1": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "PAY_2": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "PAY_3": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "PAY_4": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "PAY_5": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "PAY_6": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            }
        },
        "target": "def. pay. n. m.",  # the  label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        "model": {
            "class": LogisticRegression,
            "kwargs": {
                "class_weight": None,  # class_weight: dict or ‘balanced’, default=None
                "max_iter": 1200,  # max_iter: int, default=100
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "eeg": {
        "name": "EEG Eye State Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State",
        "header": [0],
        "drop_cols": [],  # columns to drop
        "categorical_vars": None,  # None --> NO categorical (i.e., discrete) variables (i.e., features)
        "target": "Eye Detection",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        "model": {
            "class": KNeighborsClassifier,
            "kwargs": {
                "n_neighbors": 5,
                "weights": 'uniform',  # weights: {‘uniform’, ‘distance’} or callable, default=’uniform’
                "algorithm": 'auto',  # algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                "p": 2,  # p: int, default=2 (p = 1 --> l1, manhattan; p = 2 --> l2, euclidean)
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "iris": {
        "name": "Iris Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/iris",
        "header": [0],
        "drop_cols": [],  # columns to drop
        "categorical_vars": None,  # None --> NO categorical (i.e., discrete) variables (i.e., features)
        "target": "class",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        "model": {
            "class": KNeighborsClassifier,
            "kwargs": {
                "n_neighbors": 5,
                "weights": 'uniform',  # weights: {‘uniform’, ‘distance’} or callable, default=’uniform’
                "algorithm": 'auto',  # algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                "p": 2,  # p: int, default=2 (p = 1 --> l1, manhattan; p = 2 --> l2, euclidean)
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "letter": {
        "name": "Letter Recognition Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/letter+recognition",
        "header": [0],
        "drop_cols": [],  # columns to drop
        "categorical_vars": None,  # None --> NO categorical (i.e., discrete) variables (i.e., features)
        "target": "letter",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        # "model": {
        #     "class": LogisticRegression,
        #     "kwargs": {
        #         "class_weight": None,    # class_weight: dict or ‘balanced’, default=None
        #         "max_iter": 300,         # max_iter: int, default=100
        #         "n_jobs": -1             # -1 means using all processors
        #     }
        # }
        # "model": {
        #     "class": SVC,
        #     "kwargs": {
        #         "C": 1000,               # C: float, default=1.0
        #         "kernel": "rbf",         # kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        #         "gamma": 0.01,           # gamma: {‘scale’, ‘auto’} or float, default=’scale’
        #         "probability": True      # probability: bool, default=False
        #     }
        # }
        "model": {
            "class": KNeighborsClassifier,
            "kwargs": {
                "n_neighbors": 10,
                "weights": 'uniform',  # weights: {‘uniform’, ‘distance’} or callable, default=’uniform’
                "algorithm": 'ball_tree',  # algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                "p": 2,  # p: int, default=2 (p = 1 --> l1, manhattan; p = 2 --> l2, euclidean)
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "mushroom_lb": {
        "name": "Mushroom Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/Mushroom",
        "header": [0],
        "drop_cols": ["stalk-root"],  # columns to drop
        "categorical_vars": {  # the cat. (i.e., discrete) vars. (i.e., features) that need to be encoded
            "cap-shape": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "cap-surface": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "cap-color": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "bruises": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "odor": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "gill-attachment": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "gill-spacing": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "gill-size": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "gill-color": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "stalk-shape": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "stalk-surface-above-ring": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "stalk-surface-below-ring": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "stalk-color-above-ring": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "stalk-color-below-ring": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "veil-type": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "veil-color": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "ring-number": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "ring-type": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "spore-print-color": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "population": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "habitat": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            }
        },
        "target": "class",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        # "model": {
        #     "class": LogisticRegression,
        #     "kwargs": {
        #         "class_weight": None,    # class_weight: dict or ‘balanced’, default=None
        #         "max_iter": 1000,        # max_iter: int, default=100
        #         "n_jobs": -1             # -1 means using all processors
        #     }
        # }
        "model": {
            "class": KNeighborsClassifier,
            "kwargs": {
                "n_neighbors": 5,
                "weights": 'uniform',  # weights: {‘uniform’, ‘distance’} or callable, default=’uniform’
                "algorithm": 'auto',  # algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                "p": 2,  # p: int, default=2 (p = 1 --> l1, manhattan; p = 2 --> l2, euclidean)
                "n_jobs": -1,  # -1 means using all processors
                "leaf_size": 30,
                "metric": 'minkowski',
                "metric_params": None
            }
        }
    },
    "news": {  # PORTUGUESE --> ALGORITMI Research Centre, UMinho, Portugal
        "name": "Online News Popularity Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/online+news+popularity",
        "header": [0],
        "drop_cols": ["url"],  # columns to drop
        "categorical_vars": None,  # None --> NO categorical (i.e., discrete) variables (i.e., features)
        "target": "shares",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        "model": {
            "class": LogisticRegression,
            "kwargs": {
                "class_weight": None,  # class_weight: dict or ‘balanced’, default=None
                "max_iter": 1000,  # max_iter: int, default=100
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "spam": {
        "name": "Spambase Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/spambase",
        "header": [0],
        "drop_cols": [],  # columns to drop
        "categorical_vars": None,  # None --> NO categorical (i.e., discrete) variables (i.e., features)
        "target": "spam",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        "model": {
            "class": LogisticRegression,
            "kwargs": {
                "class_weight": None,  # class_weight: dict or ‘balanced’, default=None
                "max_iter": 2000,  # max_iter: int, default=100
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "wine-red": {  # PORTUGUESE --> DSI, UMinho, Portugal
        "name": "Yeast Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/wine+quality",
        "header": [0],
        "drop_cols": [],  # columns to drop
        "categorical_vars": None,  # None --> NO categorical (i.e., discrete) variables (i.e., features)
        "target": "quality",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (0, 1)
        },
        # "model": {
        #     "class": LogisticRegression,
        #     "kwargs": {
        #         "class_weight": None,    # class_weight: dict or ‘balanced’, default=None
        #         "max_iter": 1000,        # max_iter: int, default=100
        #         "n_jobs": -1             # -1 means using all processors
        #     }
        # }
        "model": {
            "class": KNeighborsClassifier,
            "kwargs": {
                "n_neighbors": 5,
                "weights": 'uniform',  # weights: {‘uniform’, ‘distance’} or callable, default=’uniform’
                "algorithm": 'auto',  # algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                "p": 2,  # p: int, default=2 (p = 1 --> l1, manhattan; p = 2 --> l2, euclidean)
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "wine-white": {  # PORTUGUESE --> DSI, UMinho, Portugal
        "name": "Yeast Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/wine+quality",
        "header": [0],
        "drop_cols": [],  # columns to drop
        "categorical_vars": None,  # None --> NO categorical (i.e., discrete) variables (i.e., features)
        "target": "quality",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (0, 1)
        },
        # "model": {
        #     "class": LogisticRegression,
        #     "kwargs": {
        #         "class_weight": None,    # class_weight: dict or ‘balanced’, default=None
        #         "max_iter": 1000,        # max_iter: int, default=100
        #         "n_jobs": -1             # -1 means using all processors
        #     }
        # }
        "model": {
            "class": KNeighborsClassifier,
            "kwargs": {
                "n_neighbors": 5,
                "weights": 'uniform',  # weights: {‘uniform’, ‘distance’} or callable, default=’uniform’
                "algorithm": 'auto',  # algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                "p": 2,  # p: int, default=2 (p = 1 --> l1, manhattan; p = 2 --> l2, euclidean)
                "n_jobs": -1  # -1 means using all processors
            }
        }
    },
    "yeast": {
        "name": "Yeast Data Set",
        "url": "https://archive.ics.uci.edu/ml/datasets/Yeast",
        "header": [0],
        "drop_cols": ["sequence name"],  # columns to drop
        "categorical_vars": {
            "erl": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            },
            "pox": {
                "class": OneHotEncoder,
                "kwargs": {"dtype": int}
            }
        },
        "target": "local. site",  # the label of the dependent variable (i.e., feature)
        "scaler": {
            "class": MinMaxScaler,
            "feature_range": (-1, 1)
        },
        "model": {
            "class": KNeighborsClassifier,
            "kwargs": {}
        }
    }
}


def accuracy_and_auroc(
        algo: str, model: BaseEstimator, original_data: np.ndarray, imputed_data: np.ndarray, target: np.ndarray,
        discrete_columns: List[int], verbose: bool = False) -> Tuple[float, float]:
    score_accuracy: float = 0
    score_auroc: float = 0

    pd.DataFrame(original_data).describe()
    pd.DataFrame(imputed_data).describe()

    original_transformer: DataTransformer = DataTransformer()
    imputed_transformer: DataTransformer = DataTransformer()
    original: np.ndarray = original_transformer.fit_transform(original_data.copy(), discrete_columns=discrete_columns)
    imputed: np.ndarray = imputed_transformer.fit_transform(imputed_data, discrete_columns=discrete_columns)

    model.fit(X=original, y=target)
    score_accuracy = accuracy_score(y_true=target, y_pred=model.predict(X=imputed))
    if len(np.unique(target)) > 2:  # multiclass case
        score_auroc = roc_auc_score(
            y_true=target, y_score=model.predict_proba(X=imputed), multi_class='ovr')
    else:  # binary case
        score_auroc = roc_auc_score(
            y_true=target, y_score=model.predict_proba(X=imputed)[:, 1], multi_class='ovr')
    if verbose:
        print("accuracy_and_auroc():")
        print(f"\taccuracy score: {score_accuracy:.4f}")
        print(f"\tauroc score:    {score_auroc:.4f}")
    return score_accuracy, score_auroc


def report(args: Namespace,
           model: BaseEstimator,
           results: Dict[str, Dict[str, Dict[str, List[Union[np.ndarray, float]]]]]) -> None:
    print(f"miss rate:    {args.miss_rate}")
    print(f"batch size:   {args.batch_size}")
    print(f"alpha:        {args.alpha}")
    print(f"clip values:  ({-1 * args.clip_value}, {+1 * args.clip_value})")
    print(f"optimizer:    {args.optimizer}")
    print(f"learn. rate:  {args.learn_rate}")
    if args.optimizer == 'GDA':
        pass
    elif args.optimizer == 'RMSProp':
        print(f"decay:        {args.decay}")
        print(f"momentum:     {args.momentum}")
        print(f"epsilon:      {args.epsilon}")
    else:  # if args.optimizer == 'Adam':
        print(f"beta 1:       {args.beta_1}")
        print(f"beta 2:       {args.beta_2}")
        print(f"epsilon:      {args.epsilon}")
    print(f"# iterations: {args.n_iterations}")
    print(f"# critic:     {args.n_critic}")
    print(f"# runs:       {args.n_runs}")
    print(f"verbose:      {args.verbose}")
    print(f"model:        {model.__str__()}")
    for dataset, dataset_results in results.items():
        print(f"dataset: {dataset}")
        # print(f"\tshape:        {data_shape}")
        for algo, algo_results in dataset_results.items():
            print(f"\talgorithm: {algo}")
            print(f"\t\trmse:             {np.mean(algo_results['rmse_lst']):.4f} "
                  f"({np.std(algo_results['rmse_lst']):.4f})")
            print(f"\t\trmse list:        {algo_results['rmse_lst']}")
            print(f"\t\texec. time (s):   {np.mean(algo_results['exec_lst']):.4f} "
                  f"({np.std(algo_results['exec_lst']):.4f})")
            print(f"\t\texec. times list: {algo_results['exec_lst']}")
            print(f"\t\taccuracy:         {np.mean(algo_results['accuracy_lst']):.4f} "
                  f"({np.std(algo_results['accuracy_lst']):.4f})")
            print(f"\t\taccuracy list:    {algo_results['accuracy_lst']}")
            print(f"\t\tauroc:            {np.mean(algo_results['auroc_lst']):.4f} "
                  f"({np.std(algo_results['auroc_lst']):.4f})")
            print(f"\t\tauroc list:       {algo_results['auroc_lst']}")


def main(args: Namespace) -> None:
    algos: List[str] = [algo.strip() for algo in args.algos.split(',')]
    algos_set: Set[str] = set(['GAIN', 'SGAIN', 'WSGAIN-CP', 'WSGAIN-GP', 'CTGAN'])  # TODO: GET RID OF HARDCODED
    datasets: List[str] = [dataset.strip() for dataset in args.datasets.split(',')]
    datasets_set: Set[str] = set(
        ['breast', 'credit', 'eeg', 'iris', 'letter', 'mushroom_lb',  # TODO: GET RID OF HARDCODED
         'news', 'spam', 'wine-red', 'wine-white', 'yeast'])  # TODO: GET RID OF HARDCODED
    callables: Dict[str, Callable[[Namespace, Tuple[int, int], Dict[str, Any]], np.ndarray]] = {
        'GAIN': gain, 'SGAIN': SGAIN, 'WSGAIN-CP': WSGAIN_CP, 'WSGAIN-GP': WSGAIN_GP, 'CTGAN': CTGANSynthesizer}  # TODO: GET RID OF HARDCODED
    results: Dict[str, Dict[str, Dict[str, List[Union[np.ndarray, float]]]]]

    if algos == ['ALL']:
        algos = sorted(algos_set)
    else:
        if not set(algos).issubset(algos_set):
            raise ValueError(f"The following algorithms are NOT supported: {set(algos) - algos_set}")
    if datasets == ['ALL']:
        datasets = sorted(datasets_set)
    else:
        if not set(datasets).issubset(datasets_set):
            raise ValueError(f"The following datasets are NOT supported: {set(datasets) - datasets_set}")

    results = {dataset: {algo: {'rmse_lst': [], 'exec_lst': [], 'accuracy_lst': [], 'auroc_lst': []} for algo in algos}
               for dataset in datasets}

    for run in range(args.n_runs):
        tqdm.write(f"run: {run}")  # "helps" in long runs

        data: np.ndarray
        miss: np.ndarray
        mask: np.ndarray
        imputed_data: np.ndarray
        model: BaseEstimator
        score_accuracy: float
        score_auroc: float
        t0: float
        t1: float
        df: pd.DataFrame

        for dataset in datasets:
            tqdm.write(f"dataset: {dataset}")  # "helps" in long runs

            data, miss, mask = data_loader(data_name=dataset, miss_rate=args.miss_rate)
            # data, miss, mask, trgt = matrices_and_target(dataset=args.dataset, miss_rate=args.miss_rate)
            df = pd.read_csv(f"./datasets/{dataset}.csv")
            df[DATASETS[dataset]["target"]] = LabelEncoder().fit_transform(df[DATASETS[dataset]["target"]])
            discrete_columns: List = []

            for algo in algos:
                t0 = time()
                if algo in ['SGAIN', 'WSGAIN-CP', 'WSGAIN-GP']:
                    if DATASETS[dataset]['categorical_vars']:
                        discrete_columns = [df.columns.get_loc(column_name)
                                            for column_name in list(DATASETS[dataset]['categorical_vars'].keys())]

                    imputed_data = callables[algo](
                        data=miss,
                        algo_parameters={key.strip(): value for key, value in args.__dict__.items()},
                        discrete_columns=discrete_columns).execute()
                elif algo in ['CTGAN']:
                    if DATASETS[dataset]['categorical_vars']:
                        discrete_columns = list(DATASETS[dataset]['categorical_vars'].keys())

                    df_original = df.drop([DATASETS[dataset]["target"]], axis=1)
                    ctgan_synth = callables[algo](
                        epochs=1000
                    )
                    ctgan_synth.fit(df_original, discrete_columns)
                    sampled = ctgan_synth.sample(df.shape[0]).values
                    imputed_data = mask * np.nan_to_num(x=miss, nan=0.00) + (1 - mask) * sampled
                else:  # if algo in ['GAIN']
                    imputed_data = callables[algo](
                        data_x=miss, gain_parameters={key.strip(): value for key, value in args.__dict__.items()})
                t1 = time()
                results[dataset][algo]['rmse_lst'].append(
                    rmse_loss(ori_data=data, imputed_data=imputed_data, data_m=mask))
                results[dataset][algo]['exec_lst'].append(t1 - t0)
                model = DATASETS[dataset]["model"]["class"](**DATASETS[dataset]["model"]["kwargs"])
                score_accuracy, score_auroc = accuracy_and_auroc(
                    algo=algo,
                    model=model,
                    original_data=data, imputed_data=imputed_data, target=df[DATASETS[dataset]["target"]],
                    discrete_columns=discrete_columns,
                    verbose=False)
                results[dataset][algo]['accuracy_lst'].append(score_accuracy)
                results[dataset][algo]['auroc_lst'].append(score_auroc)

    report(args=args, model=model, results=results)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        '--algos',
        help="a csv list of the algorithms to run (e.g., 'GAIN,SGAIN,WSGAIN-CP,WSGAIN-GP')",
        # choices=['GAIN', 'SGAIN', 'WSGAIN-CP', 'WSGAIN-GP'],
        default='SGAIN',
        type=str)
    parser.add_argument(
        '--datasets',
        help="a csv list of datasets short names",
        # choices=['breast', 'cover-type', 'credit', 'eeg', 'iris', 'letter',
        #          'mushroom', 'news', 'spam', 'wine-red', 'wine-white', 'yeast'],
        default='letter',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help="missing data probability",
        default=0.2,
        type=float)
    parser.add_argument(
        '--batch_size',
        help="number of samples in mini-batch",
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',  # NOTE: the algorithms SGAIN, WSGAIN-CP, and WSGAIN-GP do NOT use this parameter,
        help='hint probability',  # it is here just because the GAIN algorithm requires the `hint_rate` parameter
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help="hyper-parameter to compute generator's loss",
        default=100,
        type=float)
    parser.add_argument(
        '--lambd',
        help="hyper-parameter to compute critic's loss",
        default=10,
        type=float)
    parser.add_argument(
        '--clip_value',
        help="clip (penalty) value",
        default=0.01,
        type=float)
    parser.add_argument(
        '--optimizer',
        help="solvers' optimizer",
        choices=['Adam', 'GDA', 'RMSProp'],
        default='Adam',
        type=str)
    parser.add_argument(
        '--learn_rate',
        help="optimizer's learning rate",
        default=1e-3,
        type=float)
    parser.add_argument(
        '--beta_1',
        help="Adam optimizer's hyper-parameter (1st moment estimates)",
        default=0.900,
        type=float)
    parser.add_argument(
        '--beta_2',
        help="Adam optimizer's hyper-parameter (2nd moment estimates)",
        default=0.999,
        type=float)
    parser.add_argument(
        '--decay',
        help="RMSProp optimizer's hyper-parameter (discounting factor for the history/coming gradient)",
        default=0.900,
        type=float)
    parser.add_argument(
        '--momentum',
        help="RMSProp optimizer's hyper-parameter (a scalar tensor)",
        default=0.000,
        type=float)
    parser.add_argument(
        '--epsilon',
        help="Adam hyper-parameter to ensure numerical stability or RMSProp hyper-parameter to avoid zero denominator",
        default=1e-08,
        type=float)
    parser.add_argument(
        '--n_iterations',
        help="number of training iterations",
        default=10000,
        type=int)
    parser.add_argument(
        '--n_critic',
        help="number of additional iterations to train the critic",
        default=5,
        type=int)
    parser.add_argument(
        '--n_runs',
        help="number of runs",
        default=3,
        type=int)
    parser.add_argument(
        '--verbose',
        help="to control verbosity",
        choices=['False', 'True'],  # `bool` type does NOT work as expected
        default='False',  # `bool` type does NOT work as expected
        type=str)  # `bool` type does NOT work as expected

    main(args=parser.parse_args())  # rock 'n roll

# python main.py --algos="GAIN,SGAIN,WSGAIN-CP,WSGAIN-GP" --datasets="iris,yeast" --miss_rate=0.2 --optimizer=GDA --learn_rate=0.001 --n_iterations=1000 --n_runs=3
