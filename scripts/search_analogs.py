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

# Suppress RDKit logs to keep the terminal clean
RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------------------
# PARAMETERS MUST MATCH THOSE IN cluster.py
# ---------------------------------------------------------------------
RADIUS = 2
N_BITS  = 1024


def smiles_to_bitvect(smiles: str):
    """
    Generate a Morgan fingerprint (radius=2, N_BITS=1024) as an ExplicitBitVect.
    Returns None if the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def bitvect_to_float_array(fp):
    """
    Convert an RDKit ExplicitBitVect fingerprint to a numpy.float32 array of length N_BITS.
    """
    arr = np.zeros((N_BITS,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def search_in_cluster(cluster_inds: list,
                      fp_filepath: str,
                      query_fp_bitvect,
                      query_fp_float: np.ndarray,
                      top_k: int = 5):
    """
    Search for analogs within the specified indices (cluster_inds).
    Steps:
      1) Memory-map the fingerprint file (float32[N, N_BITS]) in read-only mode
      2) Build candidate_matrix = fp_memmap[cluster_inds]
      3) Build a local FAISS IndexFlatL2 on candidate_matrix
      4) Retrieve top 50 by L2 distance
      5) Compute exact bitwise Tanimoto similarity on those 50 via RDKit
      6) Sort descending by Tanimoto and return top_k

    Returns: list of tuples (smiles, zinc_id, float(tanimoto))
    """
    global df_proc, N

    # 1) Open memmap in read-only mode
    fp_memmap = np.memmap(fp_filepath, dtype=np.float32, mode='r', shape=(N, N_BITS))

    # 2) Submatrix for FAISS
    candidate_matrix = fp_memmap[cluster_inds]  # shape = (len(cluster_inds), N_BITS)
    n_i = candidate_matrix.shape[0]

    # 3) FAISS L2 index
    index = faiss.IndexFlatL2(N_BITS)
    index.add(candidate_matrix)

    # 4) Search top 50 by L2
    top_k_faiss = min(50, n_i)
    D, I = index.search(query_fp_float.reshape(1, -1), top_k_faiss)

    # 5) Compute bitwise Tanimoto on candidates
    results = []
    for local_idx in I[0]:
        real_idx = cluster_inds[local_idx]
        cand_smi  = df_proc.iloc[real_idx]["smiles"]
        cand_zid  = df_proc.iloc[real_idx]["zinc_id"]
        cand_fp   = smiles_to_bitvect(cand_smi)
        if cand_fp is None:
            continue
        sim = TanimotoSimilarity(query_fp_bitvect, cand_fp)
        results.append((cand_smi, cand_zid, float(sim)))

    # 6) Sort by Tanimoto (descending) and return top_k
    results = sorted(results, key=lambda x: x[2], reverse=True)[:top_k]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast analog search in hierarchically clustered data"
    )
    parser.add_argument("cluster_dir",
                        help="Directory produced by cluster_build.py containing: ``processed.csv``, ``fp_matrix.dat``, ``clusters.json``, ``centroids.npy``")
    parser.add_argument("query_smiles",
                        help="Query SMILES, e.g. 'CCO' or a longer string")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top analogs to return (default: 5)")
    args = parser.parse_args()

    # 1) File paths from cluster_build.py
    processed_csv = os.path.join(args.cluster_dir, "processed.csv")
    fp_filepath   = os.path.join(args.cluster_dir, "fp_matrix.dat")
    clusters_json = os.path.join(args.cluster_dir, "clusters.json")
    centroids_npy = os.path.join(args.cluster_dir, "centroids.npy")

    # 2) Load processed.csv (columns: smiles, zinc_id)
    df_proc = pd.read_csv(processed_csv)
    N = len(df_proc)

    # 3) Load clusters.json -> { "0": [...], ... }
    with open(clusters_json) as f:
        clusters = json.load(f)

    # 4) Load centroids
    centroids = np.load(centroids_npy)  # shape = (n_clusters, N_BITS)

    # 5) Prepare query fingerprint
    q_bit = smiles_to_bitvect(args.query_smiles)
    if q_bit is None:
        print("✖ Invalid query SMILES; could not parse to Mol.")
        exit(1)
    q_float = bitvect_to_float_array(q_bit)

    # 6) Compute L2 distances to centroids and sort
    diffs = centroids - q_float.reshape(1, -1)
    centroid_dists = np.linalg.norm(diffs, axis=1)
    sorted_cids = list(np.argsort(centroid_dists))

    # 7) Accumulate candidate indices until at least 50 molecules
    candidate_inds = []
    clusters_used = []
    for cid in sorted_cids:
        clusters_used.append(cid)
        candidate_inds.extend(clusters[str(cid)])
        if len(candidate_inds) >= 50:
            break

    print(f"Gathering analogs from {len(clusters_used)} clusters (IDs: {clusters_used}), total {len(candidate_inds)} molecules.\n")

    # 8) Search within the gathered candidates
    results = search_in_cluster(candidate_inds, fp_filepath, q_bit, q_float, top_k=args.top_k)

    # 9) Print top_k results
    print(f"Top {args.top_k} most similar molecules (Tanimoto) among candidates:")
    for rank, (smi, zid, sim) in enumerate(results, start=1):
        print(f"{rank}. SMILES: {smi}, ZINC_ID: {zid}, Tanimoto = {sim:.4f}")