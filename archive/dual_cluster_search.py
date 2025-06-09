#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import faiss
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray, TanimotoSimilarity
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

RADIUS = 2
N_BITS = 1024


def smiles_to_bitvect(smiles: str):
    """
    Creates a Morgan bit fingerprint (radius=2, N_BITS=1024).
    Returns None if the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def bitvect_to_float_array(fp):
    """
    Converts an ExplicitBitVect (bit fingerprint) into a numpy.float32 array of length N_BITS.
    """
    arr = np.zeros((N_BITS,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def search_in_cluster(cluster_inds: list,
                      fp_filepath: str,
                      query_fp_bitvect,
                      query_fp_float: np.ndarray,
                      df_proc: pd.DataFrame,
                      N: int,
                      top_k: int = 5):
    """
    Searches for analogs within the provided indices (cluster_inds).
    1) Open the memory-mapped fingerprint file (float32[N, N_BITS]) in read-only mode.
    2) Extract the submatrix of candidates: fp_memmap[cluster_inds].
    3) Build a local FAISS IndexFlatL2 on the candidate matrix.
    4) Retrieve the top 50 (or fewer) by L2 distance.
    5) Compute the exact bitwise Tanimoto similarity for those 50 using RDKit.
    6) Sort descending by Tanimoto and return the top_k results.
    Returns a list of tuples: (smiles, zinc_id, float(tanimoto))
    """
    fp_memmap = np.memmap(fp_filepath, dtype=np.float32, mode='r', shape=(N, N_BITS))
    candidate_matrix = fp_memmap[cluster_inds]
    n_i = candidate_matrix.shape[0]

    index = faiss.IndexFlatL2(N_BITS)
    index.add(candidate_matrix)

    top_k_faiss = min(50, n_i)
    D, I = index.search(query_fp_float.reshape(1, -1), top_k_faiss)

    results = []
    for local_idx in I[0]:
        real_idx = cluster_inds[local_idx]
        cand_smi = df_proc.iloc[real_idx]["smiles"]
        cand_zid = df_proc.iloc[real_idx]["zinc_id"]
        cand_fp = smiles_to_bitvect(cand_smi)
        if cand_fp is None:
            continue
        sim = TanimotoSimilarity(query_fp_bitvect, cand_fp)
        results.append((cand_smi, cand_zid, float(sim)))

    results = sorted(results, key=lambda x: x[2], reverse=True)[:top_k]
    return results


def collect_candidates_and_search(cluster_dir: str,
                                  query_fp_bitvect,
                                  query_fp_float: np.ndarray,
                                  top_k: int):
    """
    For a single cluster directory:
    1) Load processed.csv, clusters.json, centroids.npy.
    2) Compute L2 distances from the query to each centroid.
    3) Accumulate candidate indices until at least 50 are collected or all clusters are exhausted.
    4) Call search_in_cluster and return the results.
    """
    processed_csv = os.path.join(cluster_dir, "processed.csv")
    fp_filepath = os.path.join(cluster_dir, "fp_matrix.dat")
    clusters_json = os.path.join(cluster_dir, "clusters.json")
    centroids_npy = os.path.join(cluster_dir, "centroids.npy")

    df_proc = pd.read_csv(processed_csv)
    N = len(df_proc)

    with open(clusters_json) as f:
        clusters = json.load(f)

    centroids = np.load(centroids_npy)

    diffs = centroids - query_fp_float.reshape(1, -1)
    centroid_dists = np.linalg.norm(diffs, axis=1)
    sorted_cids = list(np.argsort(centroid_dists))

    candidate_inds = []
    clusters_used = []
    for cid in sorted_cids:
        clusters_used.append(cid)
        candidate_inds.extend(clusters[str(cid)])
        if len(candidate_inds) >= 50:
            break

    print(f"[{os.path.basename(cluster_dir)}] Gathering analogs from {len(clusters_used)} clusters "
          f"(IDs: {clusters_used}), total of {len(candidate_inds)} molecules.")

    results = search_in_cluster(candidate_inds, fp_filepath,
                                query_fp_bitvect, query_fp_float,
                                df_proc, N, top_k=top_k)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search for analogs in two cluster sets and select the most similar."
    )
    parser.add_argument("cluster_dir1",
                        help="First directory created by cluster_build.py.")
    parser.add_argument("cluster_dir2",
                        help="Second directory created by cluster_build.py.")
    parser.add_argument("query_smiles",
                        help="Query SMILES string, e.g. 'CCO' or a longer sequence.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top analogs to return (default: 5).")
    args = parser.parse_args()

    q_bit = smiles_to_bitvect(args.query_smiles)
    if q_bit is None:
        print("✖ Invalid query SMILES. Could not parse into a molecule.")
        exit(1)
    q_float = bitvect_to_float_array(q_bit)

    print(f"Searching for analogs in '{args.cluster_dir1}' and '{args.cluster_dir2}' for SMILES = {args.query_smiles}\n")
    results1 = collect_candidates_and_search(args.cluster_dir1, q_bit, q_float, args.top_k)
    results2 = collect_candidates_and_search(args.cluster_dir2, q_bit, q_float, args.top_k)

    labelled_results = []
    for smi, zid, sim in results1:
        labelled_results.append((smi, zid, sim, os.path.basename(args.cluster_dir1)))
    for smi, zid, sim in results2:
        labelled_results.append((smi, zid, sim, os.path.basename(args.cluster_dir2)))

    labelled_results = sorted(labelled_results, key=lambda x: x[2], reverse=True)[:args.top_k]

    print(f"\nFinal top {args.top_k} most similar compounds (Tanimoto) from both sets:")
    for rank, (smi, zid, sim, src) in enumerate(labelled_results, start=1):
        print(f"{rank}. SMILES: {smi}, ZINC_ID: {zid}, Tanimoto = {sim:.4f}  (source: {src})")
