# ICCS 2021
Codebase for "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation"

This branch also includes a solution that uses the [CTGAN](https://github.com/sdv-dev/CTGAN/) method for synthetic data generation to impute missing data.

## Prerequisites

Install these packages required by CTGAN. (Creating a new separate environment is advised).

Install pytorch. Instructions [here](https://pytorch.org/).

````bash
conda install -c sdv-dev -c conda-forge rdt
````


## Usage Example
In order to make use of the mixed data types feature, the discrete columns must be specified in the metadata section of the datasets that are going to be used in main.py.

<pre>
python main.py --algos="GAIN,SGAIN,WSGAIN-CP,WSGAIN-GP,CTGAN" --datasets="iris,yeast" --miss_rate=0.2 
               --optimizer=GDA --learn_rate=0.001 
               --n_iterations=1000 --n_runs=3
</pre>

## Citing
<pre>
@inproceedings{neves:iccs:2021,
   title     = {{SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation}},
   author    = {Diogo Telmo Neves, Marcel Ganesh Naik, and Alberto Proença},
   booktitle = {The 20th International Conference on Computational Science (ICCS '21)},
   month     = June,
   year      = 2021
}
</pre>
