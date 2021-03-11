# ICCS 2021
Codebase for "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation"

## Usage Example
python main.py --algos="GAIN,SGAIN,WSGAIN-CP,WSGAIN-GP" --datasets="iris,yeast" --miss_rate=0.2 --optimizer=GDA --learn_rate=0.001 --n_iterations=1000 --n_runs=3

## Citing
@inproceedings{neves:iccs:2021, <br />
  title     = {{SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation}}, <br />
  author    = {Diogo Telmo Neves, Marcel Ganesh Naik, and Alberto Proença}, <br />
  booktitle = {The 20th International Conference on Computational Science (ICCS '21)}, <br />
   month     = June, <br />
  year      = 2021 <br />
}
