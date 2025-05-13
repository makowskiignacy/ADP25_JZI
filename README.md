# Similar Molecule Search in Large Chemical Databases

## Project Overview

The aim of this project is to develop and benchmark methods for searching large molecular databases (e.g., ChEMBL) to find similar compounds. The system will be tested using at least two clustering algorithms and two similarity measures. One of the similarity measures will be based on synthons, using fragment-based representations inspired by retrosynthetic analysis.

## Key Features

- Fragmentation of molecules into synthons using RDKit and custom rules
- Similarity comparison based on:
  - Traditional molecular fingerprints (e.g., Morgan, MACCS)
  - Synthons-based descriptors
- Clustering of the database to improve search speed and quality
  - At least two methods will be tested (e.g., K-means, hierarchical clustering)
- Performance evaluation on large datasets (e.g., ChEMBL)

---

## Proposed First Round

### Goal
Create fast search engine for organic molecules.

### Team Structure (4 Members)
Take one clustering method and test it on chembl: Deep clustering of small molecules
at large‑scale via variational autoencoder
embedding and K‑means.
| **Member** | **Task Title**           | **Description**            | 
| ---------- | ------------------------ | -------------------------- | 
| Ignacy     |                          |                            | 
| Ryszard    |                          |                            |
| Zuzanna    |                          |                            | 
| Julia      |                          |                            |


---

## Notes

- Use `psutil` or `memory_profiler` to measure memory usage.
- Prefer using `tqdm` or `timeit` for progress and timing.
- If performance is low, consider evaluating only a portion of the similarity matrix (e.g., top-k).
