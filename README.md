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

---

## Analog Retrieval with `search_analogs.py`

Retrieve top-K nearest neighbors by combining FAISS and exact Tanimoto:

```bash
python search_analogs.py <cluster_dir> <output_dir> <query_smiles> [--top_k 5]
```

* `<cluster_dir>`: directory produced by `cluster_build.py` containing
  `processed.csv`, `fp_matrix.dat`, `clusters.json` and `centroids.npy`
* `<output_dir>`: directory where `analogs.png` and `analog_results.csv` will be saved
* `<query_smiles>`: SMILES string to query (e.g. `CCO`)
* `--top_k`: number of analogs to return (default: 5)

**Workflow:**

1. **Load data**
   Reads

   * `processed.csv` (with SMILES & ZINC IDs)
   * `fp_matrix.dat` (float32 fingerprint memmap)
   * `clusters.json` (mapping of centroid → member indices)
   * `centroids.npy` (centroid float vectors)

2. **Fingerprint query**
   Converts your SMILES into

   * a 1024-bit Morgan fingerprint (radius = 2)
   * a corresponding `float32` array for FAISS

3. **Cluster selection**
   Computes L2 distances between the query array and each centroid, then accumulates members of the closest clusters until ≥ 50 candidates are gathered.

4. **FAISS search**
   Builds an `IndexFlatL2` over the candidate float vectors and retrieves up to 50 nearest by L2 distance.

5. **Exact scoring**
   For each FAISS hit:

   * Re-generate the bit-vector Morgan fingerprint
   * Compute **Tanimoto** and **Cosine** similarities
   * Sort by Tanimoto, take the top K

6. **Output**

   * Prints the top K SMILES/ZINC IDs with their Tanimoto & Cosine scores
   * Saves a grid image of the top K molecules (`analogs.png`) in `<output_dir>`
   * Writes a summary CSV (`analog_results.csv`) with SMILES, ZINC\_ID, tanimoto, cosine



**Precomputed Clustering:**  
If you want to search against a precomputed clustering (based on ~1 million compounds), the results are available here:  
[🔗 Google Drive – zinc_clusters_1M](https://drive.google.com/drive/folders/15KJIFM9LqTD0i5LmBmkI-H5Pwji7IYlK?usp=share_link)  
_This archive contains `processed.csv`, `fp_matrix.dat`, `clusters.json`, and `centroids.npy`, ready to be used with `search_analogs.py`._


## License & Citation

* **License:** MIT

* **Please cite:**

  * Szkóp, J., Milczarska, Z., Makowski, I., & Kobieta, R. (2025). ChemClusterPL: ZINC Tranche Clustering & Hierarchical SMILES Clustering Workflow (Version 1.0.0) [Computer software]. GitHub. https://github.com/makowskiignacy/ChemClusterPL (Accessed June 9, 2025)
  * Any downstream tools e.g. RDKit, FAISS

---

*End of README*
