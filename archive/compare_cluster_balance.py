import random
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats # For entropy

# Attempt to import from search_clusters.py
# Handle if classes or RDKIT_AVAILABLE/HDBSCAN_AVAILABLE are not found or if methods are not implemented.
try:
    from archive.search_clusters import TanimotoNNChainSearch, HDBSCANSearch, RDKIT_AVAILABLE, HDBSCAN_AVAILABLE
except ImportError:
    print("Could not import TanimotoNNChainSearch or HDBSCANSearch from search_clusters.py. Please ensure the file exists and is in the Python path.")
    TanimotoNNChainSearch = None
    HDBSCANSearch = None
    RDKIT_AVAILABLE = False
    HDBSCAN_AVAILABLE = False

if RDKIT_AVAILABLE:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
else:
    print("RDKit not available. SMILES validation and some clustering features might be limited.")

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("FAISS library not found. FAISS KMeans clustering will be skipped.")
    FAISS_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler # Optional: for feature scaling before KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Scikit-learn library not found. Scikit-learn KMeans clustering will be skipped.")
    SKLEARN_AVAILABLE = False


# Double-check HDBSCAN availability as search_clusters.py might have a placeholder
if HDBSCAN_AVAILABLE:
    try:
        import hdbscan 
    except ImportError:
        print("hdbscan library not found, though search_clusters.py indicated it might be. HDBSCANSearch may not function.")
        HDBSCAN_AVAILABLE = False
else:
    if TanimotoNNChainSearch is not None : # Only print if HDBSCANSearch was expected
        print("hdbscan library not available according to search_clusters.py or direct import. HDBSCANSearch will be skipped.")


# --- Configuration ---
CHEMBL_FILE_PATH = "chembl_35_chemreps.txt"
NUM_SMILES_SAMPLE = 1000
SMILES_COLUMN_INDEX = 1  # 0-based index for SMILES in chembl_35_chemreps.txt
DELIMITER = '\t'         # Tab delimiter for chembl_35_chemreps.txt
SKIP_HEADER = True
OUTPUT_PLOT_FILE = "cluster_balance_violin.png"
MORGAN_RADIUS = 2 # Added for fingerprinting consistency
MORGAN_NBITS = 1024 # Added for fingerprinting consistency
N_CLUSTERS_KMEANS = 10 # Number of clusters for KMeans algorithms

# --- Helper Functions ---

def load_random_smiles(file_path, num_smiles, smiles_col_idx, delimiter, skip_header):
    """Loads a random sample of SMILES strings from a delimited file."""
    all_smiles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if skip_header:
                next(f, None)
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) > smiles_col_idx:
                    smiles = parts[smiles_col_idx]
                    if smiles: # Ensure non-empty SMILES
                        all_smiles.append(smiles)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return []

    if not all_smiles:
        return []
    
    if len(all_smiles) <= num_smiles:
        return all_smiles
    return random.sample(all_smiles, num_smiles)

def validate_smiles(smiles_list):
    """Validates SMILES strings using RDKit, returns only valid ones."""
    if not RDKIT_AVAILABLE:
        print("Skipping SMILES validation as RDKit is not available.")
        return smiles_list
    
    valid_smiles = []
    invalid_count = 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            valid_smiles.append(s)
        else:
            invalid_count += 1
    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid SMILES removed from sample during validation.")
    return valid_smiles

def smiles_to_fingerprints_for_clustering(smiles_list, radius, nbits):
    """Converts a list of SMILES to a NumPy array of Morgan fingerprints."""
    if not RDKIT_AVAILABLE:
        print("Cannot generate fingerprints as RDKit is not available.")
        return None

    fps = []
    valid_smiles_for_fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            arr = np.zeros((1,), dtype=np.int8) # Temp array for ConvertToNumpyArray
            DataStructs.ConvertToNumpyArray(fp, arr) # This is not how ConvertToNumpyArray works for bitvects
            # Correct conversion for MorganFingerprintAsBitVect to a dense numpy array for clustering
            np_fp = np.array(list(fp.ToBitString()), dtype=np.float32) 
            fps.append(np_fp)
            valid_smiles_for_fps.append(smiles)
        # Silently skip invalid SMILES here as they are already filtered by validate_smiles
    if not fps:
        return None, []
    return np.array(fps, dtype=np.float32), valid_smiles_for_fps

def get_cluster_stats(clusters_dict, method_name):
    """Calculates balance statistics for a given clustering result."""
    if not clusters_dict: 
        return {
            "Method": method_name, "Num Clusters": 0, "Min Size": np.nan, "Max Size": np.nan,
            "Mean Size": np.nan, "Median Size": np.nan, "Std Dev Size": np.nan,
            "Num Singletons": 0, "Cluster Sizes": [],
            "CV of Sizes": np.nan, "Normalized Entropy": np.nan
        }

    cluster_sizes = [len(members) for members in clusters_dict.values() if members] 
    if not cluster_sizes: 
        return {
            "Method": method_name, "Num Clusters": len(clusters_dict), "Min Size": np.nan, "Max Size": np.nan,
            "Mean Size": np.nan, "Median Size": np.nan, "Std Dev Size": np.nan,
            "Num Singletons": 0, "Cluster Sizes": [],
            "CV of Sizes": np.nan, "Normalized Entropy": np.nan
        }

    num_singletons = sum(1 for size in cluster_sizes if size == 1)
    mean_size = np.mean(cluster_sizes)
    std_dev_size = np.std(cluster_sizes) if len(cluster_sizes) > 1 else 0
    cv_of_sizes = (std_dev_size / mean_size) if mean_size > 0 else np.nan

    # Normalized Entropy
    normalized_entropy = np.nan
    if len(cluster_sizes) > 1:
        counts = np.array(cluster_sizes)
        total_items = np.sum(counts)
        if total_items > 0:
            probabilities = counts / total_items
            entropy = scipy.stats.entropy(probabilities, base=2)
            max_entropy = np.log2(len(cluster_sizes))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            elif len(cluster_sizes) == 1 : # single cluster
                 normalized_entropy = 1.0 # Or 0, depending on definition for single cluster. Let's say 1 for perfect distribution into 1.
            else: # no clusters or empty clusters
                 normalized_entropy = 0.0

    return {
        "Method": method_name,
        "Num Clusters": len(cluster_sizes),
        "Min Size": np.min(cluster_sizes),
        "Max Size": np.max(cluster_sizes),
        "Mean Size": mean_size,
        "Median Size": np.median(cluster_sizes),
        "Std Dev Size": std_dev_size,
        "Num Singletons": num_singletons,
        "Cluster Sizes": cluster_sizes,
        "CV of Sizes": cv_of_sizes,
        "Normalized Entropy": normalized_entropy
    }

# --- New Clustering Methods ---

def cluster_with_faiss_kmeans(smiles_list_for_clustering, n_clusters, morgan_radius, morgan_nbits):
    """Clusters SMILES using FAISS KMeans."""
    if not FAISS_AVAILABLE or not RDKIT_AVAILABLE:
        print("FAISS or RDKit not available, skipping FAISS KMeans.")
        return {}

    fingerprints, valid_smiles_for_fps = smiles_to_fingerprints_for_clustering(smiles_list_for_clustering, morgan_radius, morgan_nbits)
    if fingerprints is None or fingerprints.shape[0] < n_clusters:
        print("Not enough valid fingerprints to perform FAISS KMeans or no fingerprints generated.")
        return {}

    d = fingerprints.shape[1]  # Dimension of fingerprints
    kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=20, verbose=False, gpu=False) # gpu=False for faiss-cpu
    kmeans.train(fingerprints)
    _D, I = kmeans.index.search(fingerprints, 1) # Get cluster assignment for each fingerprint

    clusters_faiss = {i: {} for i in range(n_clusters)}
    for i, smiles_idx in enumerate(I.flatten()): # I will be cluster id
        original_smiles = valid_smiles_for_fps[i]
        clusters_faiss[smiles_idx][original_smiles] = {} # Store original SMILES
    
    # Filter out empty clusters that might result if k > number of unique points after training
    return {k: v for k, v in clusters_faiss.items() if v}


def cluster_with_sklearn_kmeans(smiles_list_for_clustering, n_clusters, morgan_radius, morgan_nbits):
    """Clusters SMILES using Scikit-learn KMeans."""
    if not SKLEARN_AVAILABLE or not RDKIT_AVAILABLE:
        print("Scikit-learn or RDKit not available, skipping Scikit-learn KMeans.")
        return {}

    fingerprints, valid_smiles_for_fps = smiles_to_fingerprints_for_clustering(smiles_list_for_clustering, morgan_radius, morgan_nbits)
    if fingerprints is None or fingerprints.shape[0] < n_clusters:
        print("Not enough valid fingerprints to perform Scikit-learn KMeans or no fingerprints generated.")
        return {}

    # Optional: Scale features for KMeans - Morgan fingerprints are binary, so scaling might not be as crucial
    # scaler = StandardScaler()
    # scaled_fingerprints = scaler.fit_transform(fingerprints)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(scaled_fingerprints)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(fingerprints)
    labels = kmeans.labels_

    clusters_sklearn = {i: {} for i in range(n_clusters)}
    for i, label in enumerate(labels):
        original_smiles = valid_smiles_for_fps[i]
        clusters_sklearn[label][original_smiles] = {}
        
    return {k: v for k, v in clusters_sklearn.items() if v}


# --- Main Execution ---
def main():
    print(f"Loading up to {NUM_SMILES_SAMPLE} random SMILES from '{CHEMBL_FILE_PATH}'...")
    smiles_sample = load_random_smiles(CHEMBL_FILE_PATH, NUM_SMILES_SAMPLE, SMILES_COLUMN_INDEX, DELIMITER, SKIP_HEADER)

    if not smiles_sample:
        print("No SMILES loaded. Exiting.")
        return

    print(f"Loaded {len(smiles_sample)} SMILES strings. Validating...")
    smiles_sample = validate_smiles(smiles_sample)

    if not smiles_sample:
        print("No valid SMILES strings in the sample after validation. Exiting.")
        return
    print(f"Using {len(smiles_sample)} valid SMILES strings for clustering.")

    all_stats_list = []
    all_cluster_size_data_for_plot = [] 

    # Method 1: TanimotoNNChainSearch
    if TanimotoNNChainSearch and RDKIT_AVAILABLE:
        print("\n--- Running TanimotoNNChainSearch ---")
        try:
            searcher_nn = TanimotoNNChainSearch(similarity_threshold=0.7) 
            print("Clustering with TanimotoNNChainSearch...")
            start_time = time.time()
            clusters_nn = searcher_nn.cluster(smiles_sample)
            duration = time.time() - start_time
            print(f"TanimotoNNChainSearch clustering completed in {duration:.2f} seconds.")
            
            stats_nn = get_cluster_stats(clusters_nn, "TanimotoNNChain")
            all_stats_list.append(stats_nn)
            if stats_nn["Cluster Sizes"]:
                for size in stats_nn["Cluster Sizes"]:
                    all_cluster_size_data_for_plot.append({"Method": "TanimotoNNChain", "Cluster Size": size})
            print(f"Found {stats_nn['Num Clusters']} clusters.")

        except AttributeError as e:
            print(f"Error using TanimotoNNChainSearch: Method 'cluster' or other attribute may be missing or not implemented in 'search_clusters.py'. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with TanimotoNNChainSearch: {e}")
    else:
        print("\nSkipping TanimotoNNChainSearch (RDKit not available or TanimotoNNChainSearch class not imported/functional).")

    # Method 2: HDBSCANSearch
    if HDBSCANSearch and RDKIT_AVAILABLE and HDBSCAN_AVAILABLE:
        print("\n--- Running HDBSCANSearch ---")
        try:
            searcher_hdbscan = HDBSCANSearch(min_cluster_size=5, metric='jaccard') 
            print("Clustering with HDBSCANSearch...")
            start_time = time.time()
            clusters_hdbscan = searcher_hdbscan.cluster(smiles_sample)
            duration = time.time() - start_time
            print(f"HDBSCANSearch clustering completed in {duration:.2f} seconds.")

            stats_hdbscan = get_cluster_stats(clusters_hdbscan, "HDBSCAN")
            all_stats_list.append(stats_hdbscan)
            if stats_hdbscan["Cluster Sizes"]:
                for size in stats_hdbscan["Cluster Sizes"]:
                    all_cluster_size_data_for_plot.append({"Method": "HDBSCAN", "Cluster Size": size})
            print(f"Found {stats_hdbscan['Num Clusters']} clusters (note: HDBSCAN may label noise as singletons).")
        
        except AttributeError as e:
            print(f"Error using HDBSCANSearch: Method 'cluster' or other attribute may be missing or not implemented in 'search_clusters.py'. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with HDBSCANSearch: {e}")
    else:
        print("\nSkipping HDBSCANSearch (RDKit/HDBSCAN not available or HDBSCANSearch class not imported/functional).")
    
    # Method 3: FAISS KMeans
    if FAISS_AVAILABLE and RDKIT_AVAILABLE:
        print("\n--- Running FAISS KMeans ---")
        try:
            print(f"Clustering with FAISS KMeans (k={N_CLUSTERS_KMEANS})...")
            start_time = time.time()
            clusters_faiss = cluster_with_faiss_kmeans(smiles_sample, N_CLUSTERS_KMEANS, MORGAN_RADIUS, MORGAN_NBITS)
            duration = time.time() - start_time
            print(f"FAISS KMeans clustering completed in {duration:.2f} seconds.")
            
            stats_faiss = get_cluster_stats(clusters_faiss, f"FAISS KMeans (k={N_CLUSTERS_KMEANS})")
            all_stats_list.append(stats_faiss)
            if stats_faiss["Cluster Sizes"]:
                for size in stats_faiss["Cluster Sizes"]:
                    all_cluster_size_data_for_plot.append({"Method": f"FAISS KMeans (k={N_CLUSTERS_KMEANS})", "Cluster Size": size})
            print(f"Found {stats_faiss['Num Clusters']} clusters.")

        except Exception as e:
            print(f"An unexpected error occurred with FAISS KMeans: {e}")
    else:
        print("\nSkipping FAISS KMeans (FAISS or RDKit not available).")

    # Method 4: Scikit-learn KMeans
    if SKLEARN_AVAILABLE and RDKIT_AVAILABLE:
        print("\n--- Running Scikit-learn KMeans ---")
        try:
            print(f"Clustering with Scikit-learn KMeans (k={N_CLUSTERS_KMEANS})...")
            start_time = time.time()
            clusters_sklearn = cluster_with_sklearn_kmeans(smiles_sample, N_CLUSTERS_KMEANS, MORGAN_RADIUS, MORGAN_NBITS)
            duration = time.time() - start_time
            print(f"Scikit-learn KMeans clustering completed in {duration:.2f} seconds.")
            
            stats_sklearn = get_cluster_stats(clusters_sklearn, f"Scikit-learn KMeans (k={N_CLUSTERS_KMEANS})")
            all_stats_list.append(stats_sklearn)
            if stats_sklearn["Cluster Sizes"]:
                for size in stats_sklearn["Cluster Sizes"]:
                    all_cluster_size_data_for_plot.append({"Method": f"Scikit-learn KMeans (k={N_CLUSTERS_KMEANS})", "Cluster Size": size})
            print(f"Found {stats_sklearn['Num Clusters']} clusters.")

        except Exception as e:
            print(f"An unexpected error occurred with Scikit-learn KMeans: {e}")
    else:
        print("\nSkipping Scikit-learn KMeans (Scikit-learn or RDKit not available).")

    print("\nNote on dual_cluster_search.py: This script focuses on clustering algorithms from 'search_clusters.py' and direct k-means implementations.")
    print("'dual_cluster_search.py' appears to be structured for searching within pre-computed cluster data rather than")
    print("providing a standalone clustering algorithm for a raw list of SMILES, so it's not directly compared here.")

    # --- Display Results ---
    if not all_stats_list:
        print("\nNo clustering methods were successfully run or yielded results. Cannot generate report.")
        return

    print("\n--- Cluster Balance Comparison ---")
    stats_df = pd.DataFrame(all_stats_list)
    display_cols = ["Method", "Num Clusters", "Min Size", "Max Size", "Mean Size", "Median Size", "Std Dev Size", "Num Singletons", "CV of Sizes", "Normalized Entropy"]
    # Ensure all display_cols exist in stats_df to prevent KeyError
    valid_display_cols = [col for col in display_cols if col in stats_df.columns]
    
    if not valid_display_cols:
        print("No valid statistics columns to display.")
    else:
        print(stats_df[valid_display_cols].to_string(index=False, float_format="%.2f"))

    # --- Generate Violin Plot ---
    if not all_cluster_size_data_for_plot:
        print("\nNo cluster size data collected for any method. Skipping violin plot.")
        return
        
    plot_df = pd.DataFrame(all_cluster_size_data_for_plot)
    
    if plot_df.empty or "Cluster Size" not in plot_df.columns or plot_df["Cluster Size"].isnull().all() or plot_df["Cluster Size"].nunique() == 0 :
        print("\nCluster size data is empty, invalid, or contains no variation. Skipping violin plot.")
        return

    plt.figure(figsize=(10, 7))
    try:
        sns.violinplot(x="Method", y="Cluster Size", data=plot_df, inner="quartile", cut=0, scale="width")
        plt.title(f"Distribution of Cluster Sizes (Sample of {len(smiles_sample)} SMILES)")
        plt.ylabel("Cluster Size (Number of Compounds) - Log Scale")
        plt.xlabel("Clustering Method")
        plt.yscale('log') 
        plt.grid(True, which="both", ls="--", alpha=0.7)
        
        plt.savefig(OUTPUT_PLOT_FILE)
        print(f"\nViolin plot saved to {OUTPUT_PLOT_FILE}")
        # plt.show() 
    except Exception as e:
        print(f"Error generating or saving violin plot: {e}")
        print("Plotting may have failed due to issues with the data (e.g., all clusters same size, too few points).")

if __name__ == "__main__":
    main()
