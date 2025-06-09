## ZINC Tranche Clustering & Hierarchical SMILES Clustering Workflow

A comprehensive pipeline for downloading, extracting, clustering, and querying compounds from the ZINC database. This README integrates:

* **ZINC Tranche Download & SMILES Extraction**
* **On-Disk Hierarchical Clustering**
* **Fast Analog Retrieval (search\_analogs.py)**

---

### Table of Contents

1. [Overview](#overview)
2. [Background](#background)
3. [Prerequisites](#prerequisites)
4. [Downloading ZINC Tranches](#downloading-zinc-tranches)
5. [Extracting SMILES & ZINC IDs](#extracting-smiles--zinc-ids)
6. [Directory Structure](#directory-structure)
7. [Clustering Pipeline](#clustering-pipeline)

   * [Compute Properties & Binning](#compute-properties--binning)
   * [Hierarchical On-Disk Clustering](#hierarchical-on-disk-clustering)
8. [Analog Retrieval with `search_analogs.py`](#analog-retrieval-with-search_analogspy)
9. [License & Citation](#license--citation)

---

## Overview

This workflow streamlines the quantitative screening of commercially available compounds from the ZINC database by:

* Downloading user-selected tranches of 2D structures
* Extracting SMILES and ZINC identifiers
* Clustering compounds into physicochemical bins and creating hierarchical clusters at scale
* Providing a fast analog search tool to retrieve closest neighbors by Tanimoto similarity

It is designed for virtual screening campaigns, lead identification, scaffold hopping, and similarity searching in large libraries.

## Background

**ZINC** ([zinc.docking.org](http://zinc.docking.org)) is a freely accessible repository of purchasable compounds. It supports filtering by reactivity, purchasability, and properties such as molecular weight and log P.

Our workflow groups compounds first by user-defined physicochemical bins (e.g., MW, log P) and then applies on-disk hierarchical clustering using FAISS to handle millions of molecules without excessive RAM usage.

## Prerequisites

* **Unix-like OS** (Linux or macOS recommended)
* **Python 3.7+**
* **RDKit** (for molecular processing)
* **FAISS (cpu)** (for scalable clustering)
* **NumPy**
* **Pandas**

Install Python dependencies with:

```bash
pip install numpy pandas faiss-cpu
# RDKit installation varies by platform; see https://www.rdkit.org/docs/Install.html
```

## Downloading ZINC Tranches

1. **Obtain the downloader script**

   * From the ZINC website, select desired tranche(s) and download `ZINC-downloader-2D-txt.wget`.
   * Place it in your working directory.

2. **Make it executable & run**

   ```bash
   chmod +x ZINC-downloader-2D-txt.wget
   ./ZINC-downloader-2D-txt.wget
   ```

   This creates folders `A/`, `B/`, …, `Z/`, each containing tab-delimited text files with SMILES and metadata.

## Extracting SMILES & ZINC IDs

Use the provided Python script to consolidate SMILES and IDs:

```bash
python3 extract_zinc_smiles.py \
  --input-dir path/to/downloaded/tranche \
  --output-file zinc_smiles.txt
```

* `--input-dir`: root directory with subfolders (A/, B/, …)
* `--output-file`: path for the output txt (default `zinc_smiles.txt`)

**Output format (`zinc_smiles.txt`):**

```
<SMILES_string> <ZINC_ID>
```

Example:

```
O=C(CN1CCN(CC(=O)NC2CC2)CC1)Nc1cccc(S(=O)(=O)/N=C2/CCCN2)c1 23300202
```

## Directory Structure

After downloading and extraction, your project tree should look like:

```
.
├── ZINC-downloader-2D-txt.wget
├── extract_zinc_smiles.py
├── zinc_smiles.txt
└── <optional_workdir>/
    ├── A/
    ├── B/
    └── …
```

## Clustering Pipeline

### Compute Properties & Binning

1. **(Optional)** Compute or parse physicochemical properties (MW, log P) for each SMILES.
2. **Filter** compounds into user-defined bins, e.g. MW 200–300 Da, log P 1.0–2.0.

### Hierarchical On-Disk Clustering

Use `hierarchical_cluster.py` to build scalable clusters:

```bash
python hierarchical_cluster.py zinc_smiles.txt <output_dir>
```

* **Inputs:**

  * `zinc_smiles.txt` (TSV with header `smiles    zinc_id`)
* **Outputs in `<output_dir>`:**

  * `fp_matrix.dat`: memmapped float32 fingerprint matrix (N × 1024)
  * `clusters.json`: mapping cluster IDs → list of record indices
  * `centroids.npy`: cluster centroid vectors
  * `processed.csv`: table of valid SMILES & ZINC IDs

**Process:**

1. Reads & filters invalid SMILES
2. Generates 1024-bit Morgan fingerprints (radius = 2) in chunks to a NumPy memmap
3. Determines cluster hierarchy (√N clusters, log₂ levels)
4. Recursively runs FAISS K-means on sub-chunks from the memmap
5. Saves assignments, computes centroids

## Analog Retrieval with `search_analogs.py`

Retrieve top-K nearest neighbors by combining FAISS and exact Tanimoto:

```bash
python search_analogs.py <cluster_dir> <query_smiles> [--top_k 10]
```

* `<cluster_dir>`: output directory from `hierarchical_cluster.py`
* `<query_smiles>`: SMILES string to query (e.g. `CCO`)
* `--top_k`: number of analogs to return (default 5)

**Workflow:**

1. Loads `processed.csv`, `clusters.json`, `centroids.npy`.
2. Converts query SMILES to 1024-bit Morgan fingerprint & float32 vector.
3. Selects clusters by L2 distance to centroids until \~50 candidates collected.
4. Runs FAISS L2 search on candidate float vectors.
5. Computes bitwise Tanimoto similarity for top hits, sorts, and prints top K:

```
1. SMILES: ..., ZINC_ID: ..., Tanimoto = 0.XXXX
2. …
```

## License & Citation

* **License:** MIT

* **Please cite:**

  * Irwin, J. J., Sterling, T., Mysinger, M. M., Bolstad, E. S., & Coleman, R. G. (2012). ZINC: A free tool to discover chemistry for biology. *Journal of Chemical Information and Modeling*, **52**(7), 1757–1768.
  * Any downstream tools e.g. RDKit, FAISS

---

*End of README*
