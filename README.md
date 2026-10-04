
#GCNLA

<p align="center">
<img src="./data/GCNLA/over.png" width="80%" height="80%">
<hr></hr>
</p>



## GCNLA overview
GCNLA is used to infer cell-cell interactions based on transcriptomics data and spatial location information.  In this work, we propose a network architecture based on graph convolution network and long short-term memory attention module-GCNLA, which contains a graph convolution layer, a long short-term memory network, an attention module, and residual connections.

## Installation & Setup

Make sure to clone this repository along with the submodules as follows:

```
git clone --recurse-submodules https://github.com/sharonycc/GCNLA
cd GCNLA
```
To install dependencies to a conda environment, follow the instructions provided:
First, create a basic conda environment with python 3.8.18
```
conda create --name GCNLA python=3.8.18
```

Now, activate your environment and utilize the requirements.txt file to install non pytorch dependencies
```
conda activate GCNLA
pip install -r requirements.txt
```


## Data
Three datasets were utilized for evaluation:

1. seqFISH profile of mouse visual cortex [(Zhu _et al_., 2018)](https://www.nature.com/articles/nbt.4260)
2. MERFISH profile of mouse hypothalamic preoptic region [(Moffitt _et al_., 2018)](https://www.science.org/doi/10.1126/science.aau5324)

All of the preprocessed data are organized into pandas dataframes and are located at [./data](./data). These dataframes can be used directly as input to GCNLA.


## Run GCNLA

To run Clarify, run main.py and configure parameters based on their definitions below:

```
usage: main.py [-h] [-m MODE] [-i INPUTDIRPATH] [-o OUTPUTDIRPATH] [-s STUDYNAME] [-t SPLIT] 
               [-n NUMGENESPERCELL] [-k NEARESTNEIGHBORS] [-l LRDATABASE] [--fp FP] [--fn FN] [-a OWNADJACENCYPATH]
```
The first row of parameters are necessary

*  `-m MODE, --mode MODE`  clarify mode: preprocess,train (pick one or both separated by a comma)
*  `-i INPUTDIRPATH, --inputdirpath` Input directory path where ST dataframe is stored
*  `-o OUTPUTDIRPATH, --outputdirpath` Output directory path where results will be stored
*  `-s STUDYNAME, --studyname` clarify study name to act as identifier for outputs
*  `-t SPLIT, --split` ratio of test edges [0,1)

This second row of parameters have defaults set and are not needed.

*  `-n NUMGENESPERCELL, --numgenespercell` Number of genes in each gene regulatory network (default 45)
*  `-k NEARESTNEIGHBORS, --nearestneighbors` Number of nearest neighbors for each cell (default 5)
*  `-l LRDATABASE, --lrdatabase` 0/1/2 for which Ligand-Receptor Database to use (default 0 corresponds to mouse DB)
*  `--fp FP`               (experimentation only) add # of fake edges to train set [0,1)
*  `--fn FN`               (experimentation only) remove # of real edges from train set [0,1)
*  `-a OWNADJACENCYPATH, --ownadjacencypath` Using your own cell level adjacency (give path)

For example, if you wanted to run Clarify (both preprocessing and training) on the seqFISH data input with a 70/30 train-test split, then use the following command and set the output folder and studyname accordingly:
```
python main.py -m preprocess,train -i ./data/MERFISH/merfish_dataframe.csv -o [OUTPUT FOLDER PATH] -s [STUDYNAME] -t 0.3 
```

