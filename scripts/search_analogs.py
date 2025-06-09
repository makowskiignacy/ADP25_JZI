#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import faiss
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray, TanimotoSimilarity
from rdkit.Chem.Draw import MolsToGridImage
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
                      top_k: int = 5,
                      output_dir: str = None):
    """
    Search for analogs within the specified indices, compute tanimoto and cosine,
    and generate a single grid image with annotations.

    Returns: list of dicts with keys: smiles, zinc_id, tanimoto, cosine
    """
    global df_proc, N

    # 1) Open memmap in read-only mode
    fp_memmap = np.memmap(fp_filepath, dtype=np.float32, mode='r', shape=(N, N_BITS))

    # 2) Submatrix for FAISS
    candidate_matrix = fp_memmap[cluster_inds]
    n_i = candidate_matrix.shape[0]

    # 3) FAISS L2 index
    index = faiss.IndexFlatL2(N_BITS)
    index.add(candidate_matrix)

    # 4) Search top 50 by L2
    D_sq, I = index.search(query_fp_float.reshape(1, -1), min(50, n_i))

    # 5) Compute exact metrics on candidates
    results = []
    for rank_idx, local_idx in enumerate(I[0]):
        real_idx = cluster_inds[local_idx]
        cand_smi  = df_proc.iloc[real_idx]["smiles"]
        cand_zid  = df_proc.iloc[real_idx]["zinc_id"]
        cand_fp   = smiles_to_bitvect(cand_smi)
        if cand_fp is None:
            continue
        # Tanimoto
        sim_tan = TanimotoSimilarity(query_fp_bitvect, cand_fp)
        # Cosine on float arrays
        cand_float = bitvect_to_float_array(cand_fp)
        cos_sim = float(np.dot(query_fp_float, cand_float) /
                         (np.linalg.norm(query_fp_float) * np.linalg.norm(cand_float)))
        results.append({
            "smiles": cand_smi,
            "zinc_id": cand_zid,
            "tanimoto": sim_tan,
            "cosine": cos_sim
        })

    # 6) Sort by Tanimoto (descending) and take top_k
    results = sorted(results, key=lambda x: x["tanimoto"], reverse=True)[:top_k]

    # 7) Generate grid image if output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        mols = [Chem.MolFromSmiles(r["smiles"]) for r in results]
        legends = [f"Tanimoto: {r['tanimoto']:.2f} Cosine: {r['cosine']:.2f}" for r in results]
        grid = MolsToGridImage(mols, legends=legends, molsPerRow=len(results))
        grid_path = os.path.join(output_dir, 'analogs.png')
        grid.save(grid_path)
        # Save summary CSV
        pd.DataFrame(results).to_csv(os.path.join(output_dir, 'analog_results.csv'), index=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate analog grid for query molecules"
    )
    parser.add_argument("cluster_dir",
                        help="Directory from cluster_build.py containing processed.csv, fp_matrix.dat, clusters.json, centroids.npy")
    parser.add_argument("output_dir",
                        help="Directory to save grid image and summary CSV for top analogs")
    parser.add_argument("query_smiles",
                        help="Query SMILES, e.g. 'CCO' or a longer string")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top analogs to return (default: 5)")
    args = parser.parse_args()

    # Load inputs
    processed_csv = os.path.join(args.cluster_dir, "processed.csv")
    fp_filepath   = os.path.join(args.cluster_dir, "fp_matrix.dat")
    clusters_json = os.path.join(args.cluster_dir, "clusters.json")
    centroids_npy = os.path.join(args.cluster_dir, "centroids.npy")

    df_proc = pd.read_csv(processed_csv)
    N = len(df_proc)

    with open(clusters_json) as f:
        clusters = json.load(f)

    centroids = np.load(centroids_npy)

    # Query fingerprint
    q_bit = smiles_to_bitvect(args.query_smiles)
    if q_bit is None:
        print("✖ Invalid query SMILES; could not parse to Mol.")
        exit(1)
    q_float = bitvect_to_float_array(q_bit)

    # Find nearest centroids
    centroid_dists = np.linalg.norm(centroids - q_float.reshape(1, -1), axis=1)
    sorted_cids = list(np.argsort(centroid_dists))

    # Accumulate candidates
    candidate_inds = []
    for cid in sorted_cids:
        candidate_inds.extend(clusters[str(cid)])
        if len(candidate_inds) >= 50:
            break

    print(f"Gathering analogs from {len(candidate_inds)} molecules...\n")

    # Perform search and save outputs
    results = search_in_cluster(candidate_inds,
                                fp_filepath,
                                q_bit,
                                q_float,
                                top_k=args.top_k,
                                output_dir=args.output_dir)

    # Print results
    print(f"Top {args.top_k} most similar molecules (Tanimoto & Cosine):")
    for rank, res in enumerate(results, start=1):
        print(f"{rank}. SMILES: {res['smiles']}, ZINC_ID: {res['zinc_id']} | " \
              f"Tanimoto = {res['tanimoto']:.4f}, Cosine = {res['cosine']:.4f}")
