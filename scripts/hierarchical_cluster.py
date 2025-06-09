#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import faiss
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

RADIUS = 2
N_BITS = 1024  

class HierarchicalClusterer:
    def __init__(self, n_clusters: int, levels: int):
        self.n_clusters = n_clusters
        self.levels = levels

    def cluster_on_disk(self, fp_filepath, num_records):
        """
        Perform hierarchical clustering on a memory-mapped fingerprint file.
        - fp_filepath: path to the memmap file (.dat)
        - num_records: number of records in the memmap
        Returns: dict {cluster_id: [list of indices from 0..num_records-1]}
        """
        clusters = {}
        next_cluster_id = 0

        def recursive_cluster(indices, level):
            nonlocal next_cluster_id
            # If reached max levels or too few points for further splitting
            if level >= self.levels or len(indices) <= self.n_clusters:
                cid = next_cluster_id
                clusters[cid] = indices.copy()
                next_cluster_id += 1
                return cid

            # Read only the subset needed into RAM
            sub_matrix = fp_memmap[indices]  # shape: (len(indices), N_BITS)
            k = min(self.n_clusters, len(indices))
            kmeans = faiss.Kmeans(sub_matrix.shape[1], k, niter=20)
            kmeans.train(sub_matrix)
            _, I = kmeans.index.search(sub_matrix, 1)

            for cnum in range(k):
                sub_inds = [indices[i] for i in range(len(indices)) if I[i] == cnum]
                if not sub_inds:
                    continue
                recursive_cluster(sub_inds, level + 1)
            return None

        # Open memmap in read-only mode
        global fp_memmap
        fp_memmap = np.memmap(fp_filepath, dtype=np.float32, mode='r', shape=(num_records, N_BITS))

        recursive_cluster(list(range(num_records)), 0)
        return clusters


def smiles_to_fp_float(smi):
    """
    Convert a SMILES string to a float32 numpy array of length N_BITS.
    If the SMILES is invalid (MolFromSmiles returns None), returns None.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bitv = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    arr = np.zeros((N_BITS,), dtype=np.float32)
    ConvertToNumpyArray(bitv, arr)
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Build hierarchical clusters on a large set of SMILES (chunked)."
    )
    parser.add_argument(
        "input_txt",
        help="TSV file with header: 'smiles\tzinc_id'"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save the output files"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load the entire TSV file
    print("Loading SMILES and ZINC_ID file...")
    df = pd.read_csv(args.input_txt, sep="\t", dtype=str)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    N_total = len(df)
    print(f"Total rows in file: {N_total}.")

    # 2) Filter out invalid SMILES upfront
    print("Checking SMILES validity and filtering out invalid entries...")
    good_indices = []
    for idx, smi in enumerate(df["smiles"]):
        if Chem.MolFromSmiles(smi) is not None:
            good_indices.append(idx)
    df = df.iloc[good_indices].reset_index(drop=True)
    N = len(df)
    print(f"Valid SMILES to process: {N} (discarded {N_total - N}).")

    # 3) Create a memmap on disk: float32 matrix of shape (N, N_BITS)
    fp_filepath = os.path.join(args.output_dir, "fp_matrix.dat")
    print(f"Creating memmap for {N} × {N_BITS} floats at: {fp_filepath}...")
    fp_memmap = np.memmap(fp_filepath, dtype=np.float32, mode='w+', shape=(N, N_BITS))

    # 4) Fill the memmap in chunks of 100000
    chunk_size = 100_000
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        for idx in range(start, end):
            smi = df.iloc[idx]["smiles"]
            arr = smiles_to_fp_float(smi)
            # arr will never be None because invalid SMILES were filtered out
            fp_memmap[idx, :] = arr
        print(f"  Saved fingerprints for rows {start}–{end}")

    # Ensure data is written to disk
    del fp_memmap

    # 5) Compute hierarchy parameters
    n_clusters = int(np.sqrt(N))
    if n_clusters < 2:
        n_clusters = 2
    levels = int(np.ceil(np.log2(N / n_clusters))) if N > n_clusters else 1
    if levels < 1:
        levels = 1
    print(f"Hierarchy settings: n_clusters={n_clusters}, levels={levels} (N_BITS={N_BITS})")

    # 6) Run hierarchical clustering on the memmap
    print("Running HierarchicalClusterer on the memmap...")
    clusterer = HierarchicalClusterer(n_clusters=n_clusters, levels=levels)
    clusters = clusterer.cluster_on_disk(fp_filepath, N)  # returns {cid: [list of indices]}

    # 7) Save clusters.json
    clusters_path = os.path.join(args.output_dir, "clusters.json")
    clusters_json = {str(cid): inds for cid, inds in clusters.items()}
    with open(clusters_path, "w") as f:
        json.dump(clusters_json, f, indent=2)
    print(f"Saved clusters.json ({len(clusters_json)} clusters).")

    # 8) Compute centroids for each cluster and save to centroids.npy
    print("Computing centroids for each cluster...")
    fp_memmap = np.memmap(fp_filepath, dtype=np.float32, mode='r', shape=(N, N_BITS))

    centroids = np.zeros((len(clusters_json), N_BITS), dtype=np.float32)
    for cid_str, inds in clusters_json.items():
        cid = int(cid_str)
        sub = fp_memmap[inds]  # only loads needed subset into RAM
        centroid_vec = np.mean(sub, axis=0).astype(np.float32)
        centroids[cid] = centroid_vec

    centroids_path = os.path.join(args.output_dir, "centroids.npy")
    np.save(centroids_path, centroids)
    del fp_memmap
    print(f"Saved centroids.npy (shape = {centroids.shape}).")

    # 9) Save a minimal table: SMILES and ZINC_ID for downstream lookup
    proc_df = df[["smiles", "zinc_id"]].copy()
    proc_path = os.path.join(args.output_dir, "processed.csv")
    proc_df.to_csv(proc_path, index=False)
    print(f"Saved processed.csv at {proc_path}.")

    print("Clustering complete.")

if __name__ == "__main__":
    main()
