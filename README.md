# CommGAT


CommGAT is a graph attention framework for inferring ligand-receptor-mediated cell-cell communication from single-cell-resolution spatial transcriptomic data.

CommGAT can be applied to both two-dimensional spatial transcriptomic data and aligned consecutive tissue sections for three-dimensional CCC inference.

---


## Repository Structure

```text
CommGAT/
│
├── run_CommGAT.py
├── CCC_gat.py
├── GATv2Conv_CommGAT.py
├── epoch_eval_utils.py
├── postprocess_CommGAT.py
├── preprocess.ipynb
├── requirements.txt
└── README.md
```

The following directories are generated or provided by the user:

```text
CommGAT/
│
├── data/
│   ├── Xenium/
│   ├── TF/
│   └── LR/
│
├── outp/
│   ├── input_graph/
│   └── metadata/
│
├── model/
├── embedding_data/
└── output/
```

Raw spatial transcriptomic datasets are **not included in this repository** and should be downloaded from their original sources.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sharonycc/CommGAT
cd CommGAT
```

Create a conda environment:

```bash
conda create -n commgat python=3.8
conda activate commgat
```

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn anndata scanpy h5py qnorm pyarrow
```

PyTorch and PyTorch Geometric should be installed according to the CUDA version available on your machine.

For example, after installing a compatible PyTorch version:

```bash
pip install torch-geometric
```

The main Python dependencies include:

```text
numpy
pandas
scipy
scikit-learn
anndata
scanpy
h5py
qnorm
torch
torch-geometric
pyarrow
```

---

## Input Data

CommGAT requires the following information:

- Spatial gene expression matrix
- Cell spatial coordinates
- Cell identifiers
- Cell-type annotations
- Ligand-receptor database
- TF-target regulatory database

For the Xenium example, the preprocessing notebook expects an AnnData object containing gene expression and spatial information.

An example input file is:

```text
data/Xenium/RSC_Xenium.h5ad
```

The LR database and TF-target database should also be prepared before preprocessing.

Example directory structure:

```text
data/
└── LR/
    └── Cell_mouse_database.csv
└── TF/
    └── mouse_tf_target.csv



```

The Xenium preprocessing implementation currently uses:

```text
Cell_class
```

as the cell-type annotation and:

```text
obsm["spatial"]
```

as the spatial coordinates.

Users applying CommGAT to other datasets should modify these fields accordingly.

---

## Data Preprocessing

The Xenium preprocessing workflow is provided in:

```text
preprocess.ipynb
```



## Run CommGAT

### Single Run

CommGAT can be trained directly from the terminal using `run_CommGAT.py`.


## Consensus Communication Inference

After all independent runs have finished, use:

```text
postprocess_CommGAT.py
```

to calculate consensus communication scores.



## Citation

If you use CommGAT in your research, please cite:

```text
CommGAT: Prior-Informed Self-Supervised Graph Attention for Spatial Cell-Cell Communication Inference
```

Citation information will be updated upon publication.

---

## Contact

For questions or issues regarding CommGAT, please open an issue in this repository.
