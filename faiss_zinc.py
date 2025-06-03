import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs import ConvertToNumpyArray, TanimotoSimilarity
import numpy as np
import faiss
import json
import time
from rdkit import RDLogger
from abc import ABC, abstractmethod
from typing import Dict, List, Union

RDLogger.DisableLog('rdApp.*')

class Clusterer(ABC):
    @abstractmethod
    def cluster(self, fp_matrix: np.ndarray) -> Dict[int, List[int]]:
        pass

class HierarchicalClusterer(Clusterer):
    def __init__(self, n_clusters: int = 100, levels: int = 2):
        self.n_clusters = n_clusters
        self.levels = levels

    def cluster(self, fp_matrix: np.ndarray) -> Dict[int, List[int]]:
        clusters = {}
        next_cluster_id = 0

        def recursive_cluster(indices: List[int], level: int, parent_id: int = None) -> int:
            nonlocal next_cluster_id

            if level >= self.levels or len(indices) <= self.n_clusters:
                cluster_id = next_cluster_id
                clusters[cluster_id] = indices
                next_cluster_id += 1
                return cluster_id

            k = min(self.n_clusters, len(indices))
            kmeans = faiss.Kmeans(fp_matrix.shape[1], k, niter=20)
            kmeans.train(fp_matrix[indices])
            _, I = kmeans.index.search(fp_matrix[indices], 1)

            for cluster_num in range(k):
                sub_indices = [indices[i] for i in range(len(indices)) if I[i] == cluster_num]
                if not sub_indices:
                    continue
                recursive_cluster(sub_indices, level + 1, parent_id)

            return parent_id if parent_id is not None else 0

        recursive_cluster(list(range(len(fp_matrix))), 0)
        return clusters

# --- Parametry ---
file_path = 'zinc_base_1.txt'
clusters_path = 'hierarchical_clusters_zinc.json'
n_samples = 1000
n_bits = 2048
radius = 2
top_k_faiss = 50
top_k_tanimoto = 5
query_smiles = "COc1ccc(cc1)C(=O)Nc2ncc(s2)C3CC3"

# --- Funkcje ---
def smiles_to_bitvect(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def bitvect_to_float_array(fp):
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr

def save_clusters(clusters: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(clusters, f)

def load_clusters(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

# --- Pomiar czasu ogólnego ---
total_start = time.time()

# --- Wczytywanie danych ---
print("Wczytywanie danych...")
start = time.time()
usecols = ['smiles', 'zinc_id']
df = pd.read_csv(file_path, sep="\t", encoding="utf-16", usecols=usecols)
df = df.dropna(subset=['smiles']).sample(n=n_samples, random_state=42).reset_index(drop=True)

print("Generowanie fingerprintów...")
df['fp_bitvect'] = df['smiles'].apply(smiles_to_bitvect)
df = df.dropna(subset=['fp_bitvect']).reset_index(drop=True)
df['fp_float'] = df['fp_bitvect'].apply(bitvect_to_float_array)
fp_matrix = np.stack(df['fp_float'].values)
print(f"Czas ładowania i fingerprintów: {time.time() - start:.2f} s")

# --- Klasteryzacja ---
try:
    print("Wczytywanie istniejących klastrów...")
    clusters = load_clusters(clusters_path)
    clusters = {int(k): v for k, v in clusters.items()}
    print("Znaleziono istniejące klastry.")
except FileNotFoundError:
    print("Generowanie nowych klastrów hierarchicznych...")
    start = time.time()
    clusterer = HierarchicalClusterer(n_clusters=50, levels=2)
    clusters = clusterer.cluster(fp_matrix)
    save_clusters(clusters, clusters_path)
    print(f"Czas klasteryzacji: {time.time() - start:.2f} s")

# --- Obliczanie centroidów i struktura SMILES ---
print("Obliczanie centroidów...")
start = time.time()
cluster_centroids = {}
structured_clusters = {}

for cluster_id, indices in clusters.items():
    valid_indices = [i for i in indices if i < len(df)]
    if not valid_indices:
        continue
    vectors = fp_matrix[valid_indices]
    centroid = np.mean(vectors, axis=0)
    cluster_centroids[cluster_id] = centroid
    clusters[cluster_id] = valid_indices

    # SMILES reprezentant klastra
    dists = np.linalg.norm(vectors - centroid.reshape(1, -1), axis=1)
    closest_idx = valid_indices[np.argmin(dists)]
    centroid_smiles = df.iloc[closest_idx]['smiles']

    cluster_smiles_dict = {
        df.iloc[i]['smiles']: {} for i in valid_indices
    }
    structured_clusters[centroid_smiles] = cluster_smiles_dict

print(f"Czas przetwarzania klastrów: {time.time() - start:.2f} s")

# --- Zapis struktury do pliku JSON ---
with open("structured_clusters_zinc.json", "w") as f:
    json.dump(structured_clusters, f, indent=2)
print("Zapisano plik: structured_clusters_by_smiles.json")

# --- Zapytanie ---
print(f"\nZapytanie: {query_smiles}")
query_fp_bitvect = smiles_to_bitvect(query_smiles)
if query_fp_bitvect is None:
    raise ValueError("Nieprawidłowy SMILES zapytania.")
query_fp_float = bitvect_to_float_array(query_fp_bitvect).reshape(1, -1)

# Znajdź najbliższy klaster
start = time.time()
min_dist = float('inf')
assigned_cluster = None
for cluster_id, centroid in cluster_centroids.items():
    dist = np.linalg.norm(query_fp_float - centroid.reshape(1, -1))
    if dist < min_dist:
        min_dist = dist
        assigned_cluster = cluster_id
print(f"Związek przypisano do klastra: {assigned_cluster}")

# FAISS + Tanimoto
candidate_indices = clusters[assigned_cluster]
if not candidate_indices:
    print("Brak związków w przypisanym klastrze.")
    exit()

candidate_matrix = fp_matrix[candidate_indices]
local_index = faiss.IndexFlatL2(n_bits)
local_index.add(candidate_matrix)
D, I = local_index.search(query_fp_float, min(top_k_faiss, len(candidate_indices)))

results = []
for i in I[0]:
    real_idx = candidate_indices[i]
    candidate_fp = df.iloc[real_idx]['fp_bitvect']
    sim = TanimotoSimilarity(query_fp_bitvect, candidate_fp)
    results.append((real_idx, sim))

results = sorted(results, key=lambda x: x[1], reverse=True)
results_top5 = results[:top_k_tanimoto]
print(f"Czas przeszukiwania klastra: {time.time() - start:.2f} s")

# --- Wyniki ---
print(f"\nTop {top_k_tanimoto} najbardziej podobnych do {query_smiles} w klastrze {assigned_cluster}:")

query_mol = Chem.MolFromSmiles(query_smiles)
top_mols = []
legends = [f"Query\n{query_smiles}"]

for rank, (idx, sim) in enumerate(results_top5):
    smiles = df.iloc[idx]['smiles']
    zinc_id = df.iloc[idx]['zinc_id']
    mol = Chem.MolFromSmiles(smiles)
    top_mols.append(mol)
    legends.append(f"{rank + 1}. {zinc_id}\nTanimoto={sim:.2f}")

img = Draw.MolsToGridImage(
    [query_mol] + top_mols,
    molsPerRow=3,
    subImgSize=(300, 300),
    legends=legends,
    useSVG=False
)

try:
    from IPython.display import display
    display(img)
except ImportError:
    img.save("top5_similar_molecules_zinc.png")
    print("\nZapisano obraz do: top5_similar_molecules_zinc.png")

for rank, (idx, sim) in enumerate(results_top5):
    print(f"{rank + 1}. SMILES: {df.iloc[idx]['smiles']}, ID: {df.iloc[idx]['zinc_id']}, Tanimoto: {sim:.4f}")

print(f"\n⏱️ Całkowity czas zapytania: {time.time() - total_start:.2f} s")
