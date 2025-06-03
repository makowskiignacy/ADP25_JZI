<!-- README.md for ZINC Tranche Clustering Workflow -->

## Overview

This repository provides a streamlined workflow for the quantitative screening and clustering of chemical compounds from the ZINC database. By leveraging predefined molecular weight and log P ranges, users can generate coherent clusters of compounds exhibiting comparable physicochemical properties. The clustering strategy is particularly useful for virtual screening campaigns, lead identification, and scaffold‐hopping initiatives.

Specifically, this README describes how to apply the clustering pipeline to additional ZINC tranches, including:

1. Downloading selected tranches.
2. Extracting SMILES strings and ZINC identifiers using a dedicated Python script.
3. Preparing the final `zinc_smiles.txt` file for downstream clustering.

---

## Table of Contents

1. [Background](#background)
2. [Prerequisites](#prerequisites)
3. [Downloading ZINC Tranches](#downloading-zinc-tranches)
4. [Extracting SMILES and ZINC IDs](#extracting-smiles-and-zinc-ids)
5. [File Structure](#file-structure)
6. [Clustering Workflow (Summary)](#clustering-workflow-summary)
7. [License and Citation](#license-and-citation)

---

## Background

The ZINC database ([http://zinc.docking.org](http://zinc.docking.org)) is a freely accessible repository of commercially available compounds. It allows filtering by:

* **Reactivity** (e.g., functional group filters or pan-assay interference compound (PAINS) filters)
* **Purchasability** (i.e., whether a compound is available for purchase)
* **Physicochemical properties** such as molecular weight and log P

In our clustering approach, compounds are grouped based on user‐defined intervals of:

* **Molecular weight**
* **log P**

This ensures that each cluster contains compounds with similar size and lipophilicity, thereby improving the consistency of subsequent virtual screening or structure‐activity relationship analyses.

---

## Prerequisites

Before running the workflow below, ensure that you have the following installed on your local machine:

* **Unix‐like operating system** (Linux or macOS recommended)
* **Python 3.7+** (with standard libraries such as `argparse`, `csv`)
* **Bash shell** (for invoking the `.wget` script)
* **`extract_zinc_smiles.py`** script (included in this repository; see Section 4)

> **Note:** The Python script `extract_zinc_smiles.py` is designed to parse each ZINC tranche’s text files and produce a single tab‐delimited file containing SMILES strings and corresponding ZINC IDs.

---

## Downloading ZINC Tranches

1. **Obtain the `.wget` script**

   * After selecting the desired ZINC tranche(s) from the ZINC website, you will receive a file named:

     ```
     ZINC-downloader-2D-txt.wget
     ```
   * Place this file into the directory where you plan to download the tranche data.

2. **Make the script executable**

   ```bash
   chmod +x ZINC-downloader-2D-txt.wget
   ```

3. **Run the downloader**

   ```bash
   ./ZINC-downloader-2D-txt.wget
   ```

   * This command will download all subfolders for the chosen tranche(s). The resulting directory structure will be organized alphabetically (e.g., folders named `A/`, `B/`, …, `Z/`), and each folder will contain multiple text files with SMILES and additional metadata.

---

## Extracting SMILES and ZINC IDs

To prepare a consolidated list of SMILES strings and ZINC identifiers for clustering, execute the following steps:

1. **Ensure the Python script is available**

   * The script `extract_zinc_smiles.py` should be located in the same directory where you ran the downloader, or you can provide its absolute path when invoking it.

2. **Run the extraction script**

   ```bash
   python3 extract_zinc_smiles.py \
     --input-dir path/to/downloaded/tranche \
     --output-file zinc_smiles.txt
   ```

   * **Arguments:**

     * `--input-dir`: Path to the root directory containing the alphabetically grouped subfolders (e.g., `A/`, `B/`, …).
     * `--output-file`: Desired name and location of the consolidated output file. By default, this will be named `zinc_smiles.txt`.

3. **Resulting `zinc_smiles.txt` format**
   Each line in `zinc_smiles.txt` will consist of two columns separated by a tab character:

   ```
   <SMILES_string>    <ZINC_ID>
   ```

   Example:

   ```
   O=C(CN1CCN(CC(=O)NC2CC2)CC1)Nc1cccc(S(=O)(=O)/N=C2/CCCN2)c1    23300202
   Cc1ccc(S(=O)(=O)OC[C@@H](OS(=O)(=O)c2ccc(C)cc2)[C@H](O)[C@H](O)[C@H](O)CO)cc1    104182481
   ```

---

## File Structure

After completing the steps in Sections 3 and 4, your directory will resemble the following:

```
.
├── ZINC-downloader-2D-txt.wget
├── extract_zinc_smiles.py
├── zing_smiles_pipeline/         ← (Optional) your working directory
│   ├── A/                        ← Alphabetical subfolder (contains multiple .txt files)
│   ├── B/                        ← Alphabetical subfolder
│   ├── …
│   └── Z/                        ← Alphabetical subfolder
└── zinc_smiles.txt              ← Consolidated SMILES ↔ ZINC_ID file
```

* **`A/`, `B/`, …, `Z/`**
  Each alphabetical subfolder contains multiple tab‐delimited text files. These files include SMILES, compound metadata (e.g., molecular weight, log P, vendor information), and other identifiers.

* **`extract_zinc_smiles.py`**
  A Python script to parse each text file recursively and write out only the SMILES string and ZINC ID to `zinc_smiles.txt`.

* **`zinc_smiles.txt`**
  The final output used for clustering. Each row has the format `<SMILES>\t<ZINC_ID>`.

---

## Clustering Workflow (Summary)

Once you have generated `zinc_smiles.txt`, the clustering workflow proceeds as follows:

1. **Compute physicochemical properties** (e.g., molecular weight, log P) for each SMILES entry (if not already provided).
2. **Filter compounds** into user‐defined bins (e.g., molecular weight between 200 and 300 Da, log P between 1.0 and 2.0).
3. **Cluster compounds** within each bin using your chosen clustering algorithm (e.g., k‐means, hierarchical clustering, or fingerprint‐based clustering).
4. **Analyze clusters** for diversity metrics (e.g., Tanimoto similarity), ensure properties are comparable within each cluster, and prioritize representative scaffolds.
5. **Select representative compounds** from each cluster for virtual screening or procurement.

> **Note:** Detailed instructions for steps 1–5 (property calculation, clustering parameters, analysis, etc.) are provided in a separate `CLUSTERING_GUIDELINES.md` file (not included here).

---

## License and Citation

* This workflow is released under the MIT License.
* If you use any portion of this pipeline in your publication, please cite:

  * The original ZINC database publication:
    Irwin, J. J., Sterling, T., Mysinger, M. M., Bolstad, E. S., & Coleman, R. G. (2012). ZINC: A free tool to discover chemistry for biology. *Journal of Chemical Information and Modeling*, **52**(7), 1757–1768.
  * Any clustering algorithm or property calculation software (e.g., RDKit) used downstream.

---

**End of README.md**
