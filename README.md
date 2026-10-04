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
├── GATv2Conv_CellNEST.py
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
│   └── LR/
│
├── database/
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
git clone <YOUR_COMM_GAT_REPOSITORY_URL>
cd CommGAT
```

Create a conda environment:

```bash
conda create -n commgat python=3.10
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
- Mouse TF-target regulatory database

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

database/
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
preprocess_xenium.ipynb
```

Before execution, specify the paths to:

```text
input h5ad file
LR database
TF-target database
```

The notebook can be launched using:

```bash
jupyter notebook preprocess_xenium.ipynb
```

Alternatively, it can be executed from the terminal:

```bash
jupyter nbconvert \
    --to notebook \
    --execute preprocess_xenium.ipynb \
    --output preprocess_xenium_executed.ipynb
```

After preprocessing, CommGAT expects the following graph files:

```text
outp/input_graph/<DATA_NAME>/<DATA_NAME>_adjacency_records2

outp/input_graph/<DATA_NAME>/<DATA_NAME>_cell_vs_gene_quantile_transformed

outp/input_graph/<DATA_NAME>/<DATA_NAME>_pseudo_labels2
```

and metadata:

```text
outp/metadata/<DATA_NAME>/<DATA_NAME>_barcode_info
```

For example:

```text
outp/input_graph/RSC_Xenium/RSC_Xenium_adjacency_records2
outp/input_graph/RSC_Xenium/RSC_Xenium_cell_vs_gene_quantile_transformed
outp/input_graph/RSC_Xenium/RSC_Xenium_pseudo_labels2

outp/metadata/RSC_Xenium/RSC_Xenium_barcode_info
```

---

## Run CommGAT

### Single Run

CommGAT can be trained directly from the terminal using `run_CommGAT.py`.

Example:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python run_CommGAT.py \
    --data_name <DATA_NAME> \
    --model_name <MODEL_NAME> \
    --run_id 1 \
    --num_epoch 5000 \
    --training_data1 ./outp/input_graph/ \
    --metadata_to ./outp/metadata/ \
    --model_path ./model/ \
    --embedding_path ./embedding_data/ \
    --hidden 512 \
    --heads 1 \
    --dropout 0 \
    --lr_rate 1e-5 \
    --manual_seed yes \
    --seed 1
```

For example, for the Xenium RSC dataset using physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python run_CommGAT.py \
    --data_name RSC_Xenium \
    --model_name CommGAT_RSC_Xenium \
    --run_id 1 \
    --num_epoch 5000 \
    --training_data1 ./outp/input_graph/ \
    --metadata_to ./outp/metadata/ \
    --model_path ./model/ \
    --embedding_path ./embedding_data/ \
    --hidden 512 \
    --heads 1 \
    --dropout 0 \
    --lr_rate 1e-5 \
    --manual_seed yes \
    --seed 1
```

To use another GPU, for example physical GPU 2:

```bash
CUDA_VISIBLE_DEVICES=2 python run_CommGAT.py \
    --data_name RSC_Xenium \
    --model_name CommGAT_RSC_Xenium \
    --run_id 1 \
    --num_epoch 5000 \
    --training_data1 ./outp/input_graph/ \
    --metadata_to ./outp/metadata/ \
    --model_path ./model/ \
    --embedding_path ./embedding_data/ \
    --hidden 512 \
    --heads 1 \
    --dropout 0 \
    --lr_rate 1e-5 \
    --manual_seed yes \
    --seed 1
```

---

## Main Training Parameters

```text
--data_name
    Dataset identifier.

--model_name
    Model identifier used for output files.

--run_id
    ID of the independent training run.

--num_epoch
    Number of training epochs.
    Default: 5000

--training_data1
    Root directory containing the preprocessed graph data.

--model_path
    Directory used to save trained model checkpoints.

--embedding_path
    Directory used to save node embeddings and attention scores.

--hidden
    Hidden dimension of the graph model.
    Default: 512

--heads
    Number of graph attention heads.
    Default: 1

--dropout
    Dropout rate.
    Default: 0

--lr_rate
    Learning rate.
    Default: 1e-5

--manual_seed
    Whether to manually specify a random seed.
    Use "yes" to enable.

--seed
    Random seed.
```

---

## Multiple Independent Runs

CommGAT uses multiple independent runs to obtain more stable communication scores.

For consensus inference, we recommend numbering runs from:

```text
1, 2, ..., N
```

rather than starting from 0.

For example, to perform 10 independent runs:

```bash
for r in $(seq 1 10)
do
    CUDA_VISIBLE_DEVICES=0 python run_CommGAT.py \
        --data_name RSC_Xenium \
        --model_name CommGAT_RSC_Xenium \
        --run_id ${r} \
        --num_epoch 5000 \
        --training_data1 ./outp/input_graph/ \
        --metadata_to ./outp/metadata/ \
        --model_path ./model/ \
        --embedding_path ./embedding_data/ \
        --hidden 512 \
        --heads 1 \
        --dropout 0 \
        --lr_rate 1e-5 \
        --manual_seed yes \
        --seed ${r}
done
```

The attention files will be saved as:

```text
embedding_data/RSC_Xenium/CommGAT_RSC_Xenium_r1_attention
embedding_data/RSC_Xenium/CommGAT_RSC_Xenium_r2_attention
...
embedding_data/RSC_Xenium/CommGAT_RSC_Xenium_r10_attention
```

---

## Consensus Communication Inference

After all independent runs have finished, use:

```text
postprocess_CommGAT.py
```

to calculate consensus communication scores.

Example:

```bash
python postprocess_CommGAT.py \
    --data_name RSC_Xenium \
    --model_name CommGAT_RSC_Xenium \
    --total_runs 10 \
    --embedding_path ./embedding_data/ \
    --metadata_from ./outp/metadata/ \
    --data_from ./outp/input_graph/ \
    --output_path ./output/ \
    --high_conf_percentile 0.95 \
    --min_top5_frequency 0.50
```

The two main consensus parameters are:

```text
--high_conf_percentile
    Minimum median percentile score required for a high-confidence
    communication edge.
    Default: 0.95

--min_top5_frequency
    Minimum fraction of independent runs in which an edge must
    occur within the top 5% of communication scores.
    Default: 0.50
```

Therefore, with the default settings, a high-confidence candidate edge must:

```text
median consensus percentile >= 0.95
```

and appear within the top 5% in at least:

```text
50% of independent runs
```

---

## Specifying Known LR Pairs

Known LR interactions can optionally be extracted during postprocessing.

For example:

```bash
python postprocess_CommGAT.py \
    --data_name RSC_Xenium \
    --model_name CommGAT_RSC_Xenium \
    --total_runs 10 \
    --embedding_path ./embedding_data/ \
    --metadata_from ./outp/metadata/ \
    --data_from ./outp/input_graph/ \
    --output_path ./output/ \
    --known_lr "Ligand1-Receptor1;Ligand2-Receptor2"
```

Multiple LR pairs should be separated by semicolons.

---

## Output Files

Consensus inference generates several output files.

```text
*_01_all_lr_edges.parquet
```

All candidate LR communication edges together with raw attention scores, percentile scores, consensus scores, spatial coordinates, and run-level statistics.

```text
*_02_cellpair_communication.parquet
```

Aggregated communication scores between sender-receiver cell pairs.

```text
*_03_high_confidence_lr_edges.csv
```

High-confidence LR communication edges identified using the consensus criteria.

```text
*_04_cell_communication_activity.csv
```

Cell-level incoming and outgoing communication activity.

```text
*_05_lr_summary.csv
```

LR-level summary statistics across all candidate communication edges.

```text
*_06_known_lr_edges.csv
```

Communication edges corresponding to user-specified known LR pairs.

```text
*_07_known_lr_summary.csv
```

Summary statistics for user-specified known LR pairs.

```text
*_08_top50_lr_for_barplot.csv
```

Top-ranked LR interactions for downstream visualization.

If Parquet support is unavailable, CommGAT automatically saves the first two outputs as compressed CSV files.

---

## Reproducibility

For reproducible independent runs, use:

```bash
--manual_seed yes --seed <SEED>
```

For example:

```bash
--manual_seed yes --seed 1
```

Different seeds can be used for the multi-run consensus procedure.

---

## Notes

The current Xenium preprocessing notebook is provided as a dataset-specific example.

For a new spatial transcriptomic dataset, users should ensure that the input data contain:

```text
gene expression
cell identifiers
spatial coordinates
cell-type annotations
```

and that gene identifiers are compatible with the LR and TF-target databases.

The graph training and postprocessing scripts are executed entirely from the command line.

---

## Citation

If you use CommGAT in your research, please cite:

```text
CommGAT: Prior-Informed Self-Supervised Graph Attention
for Spatial Cell-Cell Communication Inference
```

Citation information will be updated upon publication.

---

## Contact

For questions or issues regarding CommGAT, please open an issue in this repository.
