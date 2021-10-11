
import numpy as np
import pandas as pd

from ctgan import CTGANSynthesizer

from purify.dataset.metadata import Metadata
from purify.dataset.profiling import profiler
from purify.dataset.preprocessing import PreProcessor
from purify.encoders import get_dummies_fit_transform, get_dummies_inverse_transform
from purify.synthesizers import Synthesizer

from typing import List, Tuple


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def ctgan(dataset: str = 'adult', n_epochs: int = 10, n_samples: int = 100,
          verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # df_raw: pd.DataFrame = load_demo() if dataset == 'adulta else pd.read_csv(
    #     filepath_or_buffer=f"./datasets/{dataset}.csv", na_values='?')
    df_raw: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=f"./datasets/{dataset}.csv", skipinitialspace=True, na_values='?', skip_blank_lines=True)
    df_pre: pd.DataFrame            # pandas DataFrame to hold preprocessed data
    synthesizer: CTGANSynthesizer   # the CTGAN data synthesizer
    df_sam: pd.DataFrame            # to store the samples (i.e., the synthetic data)

    # data preprocessing
    df_pre = PreProcessor.drop_columns(df=df_raw, dataset=dataset)
    df_pre = PreProcessor.replace_miss_values_by_nans(df=df_pre, dataset=dataset)
    df_pre = PreProcessor.drop_nans(df=df_pre)
    # create the synthesizer
    synthesizer = CTGANSynthesizer(epochs=n_epochs)
    # learn from the data distribution
    synthesizer.fit(train_data=df_pre, discrete_columns=Metadata.discrete_vars(dataset=dataset, df=df_pre))
    # generate synthetic data
    df_sam = synthesizer.sample(n=n_samples)
    if verbose:
        print()
        print("--- CTGAN ---" * 3)
        print(f"dataset:       {dataset}")
        print(f"initial shape: {df_raw.shape}")
        print(f"final shape:   {df_pre.shape}")
        print(f"n_epochs:      {n_epochs}")
        print(f"n_samples:     {n_samples}")
        print("samples:")
        print(df_sam.head())
        print("...")
        print(df_sam.tail())
    return df_raw, df_sam


def iccs_2021(dataset: str = 'adult', algo: str = 'SynSGAIN', miss_rate: float = 0.2, n_samples: int = 100,
              verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # df_raw: pd.DataFrame = load_demo() if dataset == 'adult' else pd.read_csv(
    #     filepath_or_buffer=f"./datasets/{dataset}.csv")
    df_raw: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=f"./datasets/{dataset}.csv", skipinitialspace=True, na_values='?', skip_blank_lines=True)
    df_pre: pd.DataFrame    # pandas DataFrame to hold preprocessed data
    df_dum: pd.DataFrame    # pandas DataFrame to hold (NOT only) dummified data
    df_sam: pd.DataFrame    # to store the samples (i.e., the synthetic data) in a pandas DataFrame
    samples: np.ndarray     # the samples (i.e., the synthetic data)
    synthesizer: Synthesizer

    ####################################################################################################################
    # TODO: TO BE REMOVED AFTER ALL DATASETS HAVE BEEN CHARACTERIZED
    # print(profiler(df=df_raw, discrete_vars=['quality']))
    # print(df_raw.shape)
    # exit()
    ####################################################################################################################

    # data preprocessing
    df_pre = PreProcessor.drop_columns(df=df_raw, dataset=dataset)
    df_pre = PreProcessor.replace_miss_values_by_nans(df=df_pre, dataset=dataset)
    df_pre = PreProcessor.drop_nans(df=df_pre)
    # data transformation that looks like one-hot encoding
    df_dum = get_dummies_fit_transform(
        data=df_pre, discrete_cols=Metadata.discrete_vars(dataset=dataset, df=df_pre), verbose=verbose)
    synthesizer = Synthesizer(
        data=df_dum.to_numpy(),
        algo=algo,
        algo_parameters={'miss_rate': miss_rate, 'n_samples': n_samples, 'verbose': verbose})
    # sampling (i.e., get the samples)
    # samples = SynSGAIN.sampler(
    #     data=df_dum.to_numpy(),
    #     algo_parameters={'miss_rate': miss_rate, 'n_samples': n_samples, 'verbose': verbose})
    samples = synthesizer.sampler()
    # data transformation to invert (i.e., to revert) the one that looks line one-hot encoding
    df_sam = get_dummies_inverse_transform(data=pd.DataFrame(data=samples, columns=df_dum.columns),
                                           discrete_cols=Metadata.discrete_vars_and_dtypes(dataset=dataset, df=df_pre),
                                           columns_order=df_pre.columns)
    if verbose:
        print()
        print("--- ICCS 2021 ---" * 3)
        print(f"algorithm:         {algo}")
        print(f"missing rate:      {miss_rate}")
        print(f"dataset:           {dataset}")
        print(f"initial shape:     {df_raw.shape}")
        print(f"preprocess. shape: {df_pre.shape}")
        print(f"dummified shape:   {df_dum.shape}")
        print(f"n_samples:         {n_samples}")
        print("samples:")
        print(df_sam.head())
        print("...")
        print(df_sam.tail())
        print(f"\nare the pandas dataframes equals? {df_pre.equals(other=df_sam)}")
    return df_raw, df_sam


if __name__ == "__main__":
    datasets: List[Tuple[str, int]] = [
        # 0                 1                2               3              4
        ('adult', 30162), ('breast', 569), ('eeg', 14980), ('iris', 150), ('mushroom', 8124),
        # 5                   6                  7
        ('wine-red', 1599), ('wine-white', 4898), ('yeast', 1484)]
    dataset_index: int = 3
    dataset: str = datasets[dataset_index][0]
    algos: List[str] = [
        # 0           1              2
        'SynSGAIN', 'SynSGAIN-CP', 'SynSGAIN-GP']
    algo_index: int = 0
    algo: str = algos[algo_index]
    n_samples: int = datasets[dataset_index][1] * 1
    miss_rate: float = 0.50
    verbose: bool = True
    df_raw: pd.DataFrame
    df_sam: pd.DataFrame

    # df_raw, df_sam = ctgan(dataset=dataset, n_samples=n_samples, verbose=verbose)
    # report_correlations(
    #     original_data=df_raw, synthetic_data=df_sam, dataset=dataset, algorithm="CTGAN")
    df_raw, df_sam = iccs_2021(dataset=dataset, algo=algo, miss_rate=miss_rate, n_samples=n_samples, verbose=verbose)
    # report_correlations(
    #     original_data=df_raw, synthetic_data=df_sam, dataset=dataset, algorithm="GenSGAIN", miss_rate=miss_rate)
    print(profiler(df=df_sam, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_sam)))

