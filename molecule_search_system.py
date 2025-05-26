import time
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import io

# --- Helper Functions ---
def smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    """Converts SMILES string to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    return None

def load_smiles_from_tsv(filepath, smiles_column='canonical_smiles', id_column='chembl_id', max_rows=None):
    """Loads SMILES strings and their IDs from a TSV file."""
    try:
        df = pd.read_csv(filepath, sep='\t', usecols=[id_column, smiles_column], nrows=max_rows)
        # Drop rows where SMILES is NaN or None
        df.dropna(subset=[smiles_column], inplace=True)
        return df[id_column].tolist(), df[smiles_column].tolist()
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

# --- Interfaces ---
class DistanceMetric(ABC):
    @abstractmethod
    def calculate_distance(self, fp1, fp2) -> float:
        """Calculates distance between two fingerprints. Lower is more similar."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class ClusteringAlgorithm(ABC):
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = None
        self.fingerprints_in_index = None
        self.labels_in_index = None
        self.original_smiles_in_index = None
        self.ids_in_index = None # Store original IDs

    @abstractmethod
    def build_index(self, molecule_ids, smiles_list, fingerprints):
        """Builds the clustering index."""
        self.ids_in_index = molecule_ids
        self.original_smiles_in_index = smiles_list
        self.fingerprints_in_index = np.array(fingerprints) # Ensure it's a 2D numpy array

    @abstractmethod
    def search_cluster_members(self, query_fingerprint, distance_metric: DistanceMetric, n_neighbors=5) -> list:
        """Identifies relevant cluster(s) and searches within them."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

# --- Concrete Distance Metrics ---
class TanimotoDistance(DistanceMetric):
    def calculate_distance(self, fp1, fp2) -> float:
        # RDKit's TanimotoSimilarity expects RDKit fingerprint objects
        # If fp1, fp2 are numpy arrays, convert them back or use a compatible Tanimoto
        # For bit vectors (numpy arrays of 0s and 1s):
        # Tanimoto similarity = (fp1 & fp2).sum() / (fp1 | fp2).sum()
        # Distance = 1 - similarity
        if not isinstance(fp1, np.ndarray) or not isinstance(fp2, np.ndarray):
            raise ValueError("Fingerprints must be NumPy arrays for TanimotoDistance.")
        
        intersection = np.sum(np.bitwise_and(fp1.astype(bool), fp2.astype(bool)))
        union = np.sum(np.bitwise_or(fp1.astype(bool), fp2.astype(bool)))
        if union == 0:
            return 1.0 # Or 0.0 if identical empty sets are considered perfectly similar
        similarity = intersection / union
        return 1.0 - similarity
        
    def get_name(self) -> str:
        return "Tanimoto"

class DiceDistance(DistanceMetric):
    def calculate_distance(self, fp1, fp2) -> float:
        # Dice similarity = 2 * (fp1 & fp2).sum() / (fp1.sum() + fp2.sum())
        # Distance = 1 - similarity
        if not isinstance(fp1, np.ndarray) or not isinstance(fp2, np.ndarray):
            raise ValueError("Fingerprints must be NumPy arrays for DiceDistance.")

        intersection = np.sum(np.bitwise_and(fp1.astype(bool), fp2.astype(bool)))
        sum_counts = np.sum(fp1.astype(bool)) + np.sum(fp2.astype(bool))
        if sum_counts == 0:
            return 1.0 # Or 0.0
        similarity = (2.0 * intersection) / sum_counts
        return 1.0 - similarity

    def get_name(self) -> str:
        return "Dice"

# --- Concrete Clustering Algorithms ---
class KMeansClusterer(ClusteringAlgorithm):
    def __init__(self, n_clusters=5, random_state=42):
        super().__init__(n_clusters)
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')

    def build_index(self, molecule_ids, smiles_list, fingerprints):
        super().build_index(molecule_ids, smiles_list, fingerprints)
        if self.fingerprints_in_index.shape[0] < self.n_clusters:
            print(f"Warning: Number of samples ({self.fingerprints_in_index.shape[0]}) is less than n_clusters ({self.n_clusters}). Adjusting n_clusters.")
            self.model.n_clusters = self.fingerprints_in_index.shape[0]
            if self.model.n_clusters == 0: # No data
                 self.labels_in_index = np.array([])
                 return
        
        if self.fingerprints_in_index.shape[0] > 0:
            self.model.fit(self.fingerprints_in_index)
            self.labels_in_index = self.model.labels_
        else:
            self.labels_in_index = np.array([])


    def search_cluster_members(self, query_fingerprint, distance_metric: DistanceMetric, n_neighbors=5) -> list:
        if self.model is None or self.fingerprints_in_index is None or self.fingerprints_in_index.shape[0] == 0:
            return [] # Index not built or empty
        
        if self.fingerprints_in_index.shape[0] < self.model.n_clusters: # check if model was fit
            # Fallback to brute force if clustering wasn't effective (e.g. too few samples)
            # This case should ideally be handled by MoleculeSearcher or by ensuring build_index is robust
            print("KMeans: Not enough samples for effective clustering, falling back to full search within available data.")
            distances = []
            for i, fp_in_db in enumerate(self.fingerprints_in_index):
                dist = distance_metric.calculate_distance(query_fingerprint, fp_in_db)
                distances.append((self.ids_in_index[i], self.original_smiles_in_index[i], dist))
            distances.sort(key=lambda x: x[2])
            return distances[:n_neighbors]

        query_cluster_label = self.model.predict(query_fingerprint.reshape(1, -1))[0]
        
        member_indices = np.where(self.labels_in_index == query_cluster_label)[0]
        
        if len(member_indices) == 0:
            return []

        candidate_smiles = [self.original_smiles_in_index[i] for i in member_indices]
        candidate_ids = [self.ids_in_index[i] for i in member_indices]
        candidate_fingerprints = self.fingerprints_in_index[member_indices]
        
        distances = []
        for i, fp_in_cluster in enumerate(candidate_fingerprints):
            dist = distance_metric.calculate_distance(query_fingerprint, fp_in_cluster)
            distances.append((candidate_ids[i], candidate_smiles[i], dist))
            
        distances.sort(key=lambda x: x[2])
        return distances[:n_neighbors]

    def get_name(self) -> str:
        return f"KMeans(k={self.n_clusters})"

class AgglomerativeClusterer(ClusteringAlgorithm):
    def __init__(self, n_clusters=5):
        super().__init__(n_clusters)
        # Using 'euclidean' affinity as default for fingerprints,
        # 'jaccard' could be used if fingerprints are binary and represent sets.
        # However, our distance metrics (Tanimoto/Dice) are applied *after* cluster selection.
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.centroids_ = None

    def build_index(self, molecule_ids, smiles_list, fingerprints):
        super().build_index(molecule_ids, smiles_list, fingerprints)
        if self.fingerprints_in_index.shape[0] == 0: # No data
            self.labels_in_index = np.array([])
            return
        
        if self.fingerprints_in_index.shape[0] < self.n_clusters:
            # print(f"Warning: Number of samples ({self.fingerprints_in_index.shape[0]}) is less than n_clusters ({self.n_clusters}). Adjusting n_clusters for AgglomerativeClustering.")
            # AgglomerativeClustering requires n_samples >= n_clusters if linkage is not 'ward'
            # If we set n_clusters to n_samples, each point is its own cluster.
            # For simplicity, if n_samples < n_clusters, we might not cluster or adjust n_clusters.
            # Here, we let it proceed; scikit-learn might handle it or error.
            # A robust way is to cap n_clusters at n_samples.
            effective_n_clusters = min(self.n_clusters, self.fingerprints_in_index.shape[0])
            if effective_n_clusters <=1 and self.fingerprints_in_index.shape[0] > 0 : # need at least 1 cluster, or 2 for meaningful clustering
                 effective_n_clusters = self.fingerprints_in_index.shape[0] if self.fingerprints_in_index.shape[0] > 0 else 1


            self.model = AgglomerativeClustering(n_clusters=effective_n_clusters)


        self.labels_in_index = self.model.fit_predict(self.fingerprints_in_index)
        
        # Calculate centroids for assigning new query points
        self.centroids_ = []
        if hasattr(self.model, 'n_clusters_'): # scikit-learn 0.24+
            num_actual_clusters = self.model.n_clusters_
        else: # older versions might just use self.n_clusters
            num_actual_clusters = self.n_clusters
            if self.fingerprints_in_index.shape[0] < self.n_clusters: # if we adjusted
                 num_actual_clusters = effective_n_clusters


        for i in range(num_actual_clusters):
            cluster_points = self.fingerprints_in_index[self.labels_in_index == i]
            if len(cluster_points) > 0:
                self.centroids_.append(np.mean(cluster_points, axis=0))
            # else: handle empty cluster if necessary, though fit_predict usually avoids this for Agglo

    def search_cluster_members(self, query_fingerprint, distance_metric: DistanceMetric, n_neighbors=5) -> list:
        if self.model is None or self.centroids_ is None or not self.centroids_ or self.fingerprints_in_index is None or self.fingerprints_in_index.shape[0] == 0:
            return []

        if not self.centroids_: # No centroids, e.g. if build_index failed or had no data
             # Fallback to brute force
            print("Agglomerative: No centroids, falling back to full search.")
            distances = []
            for i, fp_in_db in enumerate(self.fingerprints_in_index):
                dist = distance_metric.calculate_distance(query_fingerprint, fp_in_db)
                distances.append((self.ids_in_index[i], self.original_smiles_in_index[i], dist))
            distances.sort(key=lambda x: x[2])
            return distances[:n_neighbors]

        # Find closest centroid (Euclidean distance to centroid)
        centroid_distances = [np.linalg.norm(query_fingerprint - centroid) for centroid in self.centroids_]
        if not centroid_distances: return [] # no centroids
        query_cluster_label = np.argmin(centroid_distances)
        
        member_indices = np.where(self.labels_in_index == query_cluster_label)[0]

        if len(member_indices) == 0:
            return []

        candidate_smiles = [self.original_smiles_in_index[i] for i in member_indices]
        candidate_ids = [self.ids_in_index[i] for i in member_indices]
        candidate_fingerprints = self.fingerprints_in_index[member_indices]
        
        distances = []
        for i, fp_in_cluster in enumerate(candidate_fingerprints):
            dist = distance_metric.calculate_distance(query_fingerprint, fp_in_cluster)
            distances.append((candidate_ids[i], candidate_smiles[i], dist))
            
        distances.sort(key=lambda x: x[2])
        return distances[:n_neighbors]

    def get_name(self) -> str:
        return f"Agglomerative(k={self.n_clusters})"

# --- Molecule Searcher ---
class MoleculeSearcher:
    def __init__(self, distance_metric: DistanceMetric, clustering_algorithm: ClusteringAlgorithm = None):
        self.distance_metric = distance_metric
        self.clustering_algorithm = clustering_algorithm
        
        self.all_fingerprints_in_db = [] # Raw list of fingerprints
        self.all_smiles_in_db = []       # Raw list of SMILES
        self.all_ids_in_db = []          # Raw list of IDs

        self.fp_radius = 2
        self.fp_nbits = 1024

    def _prepare_fingerprints(self, molecule_ids, smiles_list):
        valid_ids = []
        valid_smiles = []
        valid_fingerprints = []
        for mid, smiles_str in zip(molecule_ids, smiles_list):
            fp = smiles_to_morgan_fingerprint(smiles_str, radius=self.fp_radius, n_bits=self.fp_nbits)
            if fp is not None:
                valid_ids.append(mid)
                valid_smiles.append(smiles_str)
                valid_fingerprints.append(fp)
        return valid_ids, valid_smiles, valid_fingerprints

    def build_index_from_lists(self, molecule_ids: list, smiles_list: list):
        """Builds index from lists of molecule IDs and SMILES strings."""
        print(f"Building index for {len(smiles_list)} molecules...")
        start_time = time.time()
        
        self.all_ids_in_db, self.all_smiles_in_db, self.all_fingerprints_in_db = self._prepare_fingerprints(molecule_ids, smiles_list)
        
        if not self.all_fingerprints_in_db:
            print("No valid fingerprints generated. Index is empty.")
            self.fp_gen_time = time.time() - start_time
            self.index_build_time = 0
            return self.fp_gen_time, self.index_build_time

        self.fp_gen_time = time.time() - start_time # Time for fingerprint generation

        # If clustering is used, build that specific index
        cluster_build_start_time = time.time()
        if self.clustering_algorithm:
            # Pass only valid fingerprints and corresponding SMILES/IDs
            self.clustering_algorithm.build_index(self.all_ids_in_db, self.all_smiles_in_db, self.all_fingerprints_in_db)
        self.index_build_time = time.time() - cluster_build_start_time
        
        total_build_time = time.time() - start_time
        print(f"Index built. Fingerprint generation: {self.fp_gen_time:.4f}s. Clustering time: {self.index_build_time:.4f}s. Total: {total_build_time:.4f}s")
        return self.fp_gen_time, self.index_build_time


    def build_index_from_file(self, filepath, smiles_column='canonical_smiles', id_column='chembl_id', max_rows=None):
        """Loads data from file and builds the index."""
        molecule_ids, smiles_list = load_smiles_from_tsv(filepath, smiles_column, id_column, max_rows)
        if not smiles_list:
            print("No SMILES loaded from file.")
            self.fp_gen_time = 0
            self.index_build_time = 0
            return 0, 0
        return self.build_index_from_lists(molecule_ids, smiles_list)

    def search(self, query_smiles: str, n_neighbors=5) -> tuple[list, float]: # Modified return type hint
        """Searches for similar molecules."""
        if not self.all_smiles_in_db: # Check if index was built with data
            print("Searcher has no data in index. Build index first.")
            return [], 0.0 # Return tuple

        start_time = time.time()
        query_fp = smiles_to_morgan_fingerprint(query_smiles, radius=self.fp_radius, n_bits=self.fp_nbits)
        if query_fp is None:
            print(f"Could not generate fingerprint for query: {query_smiles}")
            return [], 0.0 # Return tuple

        if self.clustering_algorithm:
            results = self.clustering_algorithm.search_cluster_members(query_fp, self.distance_metric, n_neighbors)
        else: # Brute-force search
            distances = []
            for i, fp_in_db in enumerate(self.all_fingerprints_in_db):
                dist = self.distance_metric.calculate_distance(query_fp, fp_in_db)
                # Storing ID, SMILES, distance
                distances.append((self.all_ids_in_db[i], self.all_smiles_in_db[i], dist))
            
            distances.sort(key=lambda x: x[2]) # Sort by distance
            results = distances[:n_neighbors]
        
        search_time = time.time() - start_time
        # print(f"Search completed in {search_time:.6f}s")
        return results, search_time

# --- Main Execution and Benchmarking ---
if __name__ == "__main__":
    # Create a dummy TSV file for testing
    dummy_data_content = """chembl_id\tcanonical_smiles\tstandard_inchi\tstandard_inchi_key
CHEMBL153534\tCc1cc(-c2csc(N=C(N)N)n2)cn1C\tInChI=1S/C10H13N5S/c1-6-3-7(4-15(6)2)8-5-16-10(13-8)14-9(11)12/h3-5H,1-2H3,(H4,11,12,13,14)\tMFRNFCWYPYSFQQ-UHFFFAOYSA-N
CHEMBL440060\tCC[C@H](C)[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@@H](N)CCSC)[C@@H](C)O)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](CC(N)=O)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCN=C(N)N)C(=O)NCC(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N1CCC[C@H]1C(=O)N1CCC[C@H]1C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](CCCN=C(N)N)C(N)=O\tInChI=1S/C123H212N44O34S/c1-19-63(12)96(164-115(196)81(47-62(10)11)163-119(200)97(68(17)169)165-103(184)70(124)36-42-202-18)118(199)143-52-92(175)147-65(14)100(181)149-67(16)102(183)157-82(48-69-50-136-57-145-69)114(195)162-83(49-90(128)173)106(187)141-51-91(174)146-64(13)99(180)148-66(15)101(182)153-75(31-34-88(126)171)109(190)160-80(46-61(8)9)113(194)161-79(45-60(6)7)112(193)155-73(27-22-39-139-123(134)135)107(188)156-76(32-35-89(127)172)110(191)159-78(44-59(4)5)111(192)154-72(26-21-38-138-122(132)133)104(185)140-53-93(176)150-74(30-33-87(125)170)108(189)158-77(43-58(2)3)105(186)144-55-95(178)166-40-24-29-86(166)120(201)167-41-23-28-85(167)117(198)142-54-94(177)151-84(56-168)116(197)152-71(98(129)179)25-20-37-137-121(130)131/h50,57-68,70-86,96-97,168-169H,19-49,51-56,124H2,1-18H3,(H2,125,170)(H2,126,171)(H2,127,172)(H2,128,173)(H2,129,179)(H,136,145)(H,140,185)(H,141,187)(H,142,198)(H,143,199)(H,144,186)(H,146,174)(H,147,175)(H,148,180)(H,149,181)(H,150,176)(H,151,177)(H,152,197)(H,153,182)(H,154,192)(H,155,193)(H,156,188)(H,157,183)(H,158,189)(H,159,191)(H,160,190)(H,161,194)(H,162,195)(H,163,200)(H,164,196)(H,165,184)(H4,130,131,137)(H4,132,133,138)(H4,134,135,139)/t63-,64-,65-,66-,67-,68+,70-,71-,72-,73-,74-,75-,76-,77-,78-,79-,80-,81-,82-,83-,84-,85-,86-,96-,97-/m0/s1\tRSEQNZQKBMRQNM-VRGFNVLHSA-N
CHEMBL1\tCC(=O)Oc1ccccc1C(=O)OH\tInChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N
CHEMBL2\tCN1C=NC2=C1C(=O)N(C)C(=O)N2C\tInChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3\tRIYYRCRXDSANQB-UHFFFAOYSA-N
CHEMBL3\tCC(C)CC1=CC=C(C=C1)C(C)C(=O)O\tInChI=1S/C13H18O2/c1-9(2)7-11-5-3-10(4-6-11)8(12(14)15)16/h3-6,8-9H,7H2,1-2H3,(H,14,15)\tHEFNNWSXXWATRW-UHFFFAOYSA-N
CHEMBL4\tC1=CC=C(C=C1)CC(C(=O)O)N\tInChI=1S/C9H11NO2/c10-8(9(11)12)6-7-4-2-1-3-5-7/h1-5,8H,6,10H2,(H,11,12)\tCOLNVLDHVKWLRT-UHFFFAOYSA-N
CHEMBL5\tCC(=O)NC1=CC=C(C=C1)O\tInChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)\tRZERFNYDWHSVIT-UHFFFAOYSA-N
CHEMBL6\tCN(C)C1=CC=C(C=C1)N=NC2=CC=C(C=C2)S(=O)(=O)O\tInChI=1S/C14H15N3O3S/c1-16(2)12-7-3-11(4-8-12)15-10-5-9-13(14(10)17)21(18,19)20/h3-9H,1-2H3,(H,20,18,19)\tXUFQYDNAJPSQPA-UHFFFAOYSA-N
CHEMBL7\tC1COCCN1\tInChI=1S/C4H9NO/c1-2-6-4-3-5-1/h5H,1-4H2\tDEGUGPOZMYMNGP-UHFFFAOYSA-N
CHEMBL8\tCCCCCCCC(=O)O\tInChI=1S/C8H16O2/c1-2-3-4-5-6-7-8(9)10/h2-7H2,1H3,(H,9,10)\tWWZKQNGHHNEAKC-UHFFFAOYSA-N
CHEMBL10\tCC(C)(C)OC(=O)N1CCC(CC1)N\tInChI=1S/C10H20N2O2/c1-10(2,3)14-9(13)12-7-5-4-6-8(12)11/h8H,4-7,11H2,1-3H3\tLCXTAGNWSAVBFX-UHFFFAOYSA-N
CHEMBL12\tCC(C(=O)O)N\tInChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)\tQNAYBMKLOCPYGJ-UHFFFAOYSA-N
CHEMBL13\tC1=CC=C2C(=C1)C=CN2\tInChI=1S/C9H7N/c1-2-4-9-7-5-6-10-8-9/h1-2,4-5,7-8H,6H2\tJPHYBAPSAENTLZ-UHFFFAOYSA-N
CHEMBL14\tCC(=O)C1=CC=CC=C1\tInChI=1S/C8H8O/c1-7(9)8-5-3-2-4-6-8/h2-6H,1H3\tCKRZKMFTZCFYGB-UHFFFAOYSA-N
CHEMBL15\tCC(C)N\tInChI=1S/C3H9N/c1-3(2)4/h3H,4H2,1-2H3\tKDSNLYIMUZNERS-UHFFFAOYSA-N
CHEMBL16\tC1CCC(CC1)O\tInChI=1S/C6H12O/c7-6-4-2-1-3-5-6/h6-7H,1-5H2\tHPXRVTGHNJAIIH-UHFFFAOYSA-N
CHEMBL18\tC1=CC=C(C=C1)C=O\tInChI=1S/C7H6O/c8-6-7-4-2-1-3-5-7/h1-6H\tHUMNYLRZRPPJDN-UHFFFAOYSA-N
CHEMBL19\tCC(=O)NCCC(=O)O\tInChI=1S/C5H9NO3/c1-4(7)6-2-3-5(8)9/h2-3H2,1H3,(H,6,7)(H,8,9)\tWNWGYJYAENWLLJ-UHFFFAOYSA-N
CHEMBL20\tCC(C)OC(=O)C(C(C)C)N\tInChI=1S/C9H19NO2/c1-5(2)9(10)8(11)12-6(3)4/h5-6,9H,10H2,1-4H3\tGYHFUZHODSMGHL-UHFFFAOYSA-N
CHEMBL21\tO=C(NCCS)c1ccccc1\tInChI=1S/C9H11NOS/c11-9(7-4-2-1-3-5-7)10-6-8-12/h1-5,12H,6,8H2,(H,10,11)\tSNVPJSOCOMHDKP-UHFFFAOYSA-N
CHEMBL22\tCOc1cc(OC)c(C=O)c(OC)c1\tInChI=1S/C10H12O4/c1-11-7-5-6(4-14)8(12-2)10(9(7)13-3)15/h4-5H,1-3H3\tMWMHOKGMHVHBAP-UHFFFAOYSA-N
"""
    dummy_file_path = "dummy_chembl_data.tsv"
    with open(dummy_file_path, "w") as f:
        f.write(dummy_data_content)

    chembl_data_path = "chembl_35_chemreps.txt"
    # --- Benchmarking Parameters ---
    N_CLUSTERS = 3 # Keep small for tiny dataset
    N_NEIGHBORS_TO_FIND = 3
    # Use a subset of loaded molecules as queries
    # For a more robust benchmark, use a separate query set.
    # Here, we'll pick a few from the loaded data.
    
    # --- Load data ---
    # Using max_rows to keep it small and fast for this example
    # For a real benchmark, use a larger dataset (e.g., 1000s or 10000s of molecules)
    all_ids, all_smiles = load_smiles_from_tsv(chembl_data_path, max_rows=10000) # Adjust max_rows as needed for larger datasets
    if not all_smiles:
        print("No data loaded, exiting benchmark.")
        exit()
    
    print(f"Loaded {len(all_smiles)} molecules for benchmarking.")
    if len(all_smiles) < N_CLUSTERS :
        print(f"Warning: Number of molecules ({len(all_smiles)}) is less than N_CLUSTERS ({N_CLUSTERS}). Adjusting N_CLUSTERS.")
        N_CLUSTERS = max(1, len(all_smiles) -1) # Ensure at least 1 cluster, or k < n_samples
        if N_CLUSTERS == 0 and len(all_smiles) == 1: N_CLUSTERS = 1


    query_smiles_list = all_smiles[:min(3, len(all_smiles))] # Use first 3 as queries

    # --- Initialize Metrics and Algorithms ---
    tanimoto_metric = TanimotoDistance()
    dice_metric = DiceDistance()
    
    metrics = [tanimoto_metric, dice_metric]
    
    # Clustering algorithms to test
    # Ensure N_CLUSTERS is appropriate for the dataset size
    
    clustering_configs = [
        None, # For brute-force
        KMeansClusterer(n_clusters=N_CLUSTERS),
        AgglomerativeClusterer(n_clusters=N_CLUSTERS)
    ]

    benchmark_results = {
        "build_fp_time": 0,
        "build_cluster_time": [], # List of (method_name, time)
        "search_times": [] # List of (method_name, avg_search_time)
    }

    # --- Perform Benchmarking ---
    # 1. Fingerprint generation (once)
    # This is part of the first MoleculeSearcher's build_index
    
    # Temporary searcher to get fp_gen_time
    temp_searcher_for_fp_time = MoleculeSearcher(distance_metric=tanimoto_metric) # Metric choice doesn't matter here
    fp_gen_time, _ = temp_searcher_for_fp_time.build_index_from_lists(all_ids, all_smiles)
    benchmark_results["build_fp_time"] = fp_gen_time
    
    # Store all fingerprints and smiles from this initial pass to avoid recomputing
    # This assumes all_fingerprints_in_db is populated correctly by build_index_from_lists
    # and can be reused.
    # For a cleaner design, MoleculeSearcher could expose these.
    # For now, we'll rely on each searcher instance to manage its own.

    for dist_metric in metrics:
        for cluster_algo in clustering_configs:
            searcher_name_parts = [dist_metric.get_name()]
            if cluster_algo:
                searcher_name_parts.append(cluster_algo.get_name())
            else:
                searcher_name_parts.append("BruteForce")
            
            method_name = "+".join(searcher_name_parts)
            print(f"\n--- Benchmarking: {method_name} ---")

            searcher = MoleculeSearcher(distance_metric=dist_metric, clustering_algorithm=cluster_algo)
            
            # Build index
            current_fp_gen_time, current_cluster_build_time = searcher.build_index_from_lists(all_ids, all_smiles)
            # fp_gen_time is recorded once. cluster_build_time is specific to this config.
            if cluster_algo: # Only log cluster build time if a clustering algo is used
                 benchmark_results["build_cluster_time"].append((method_name, current_cluster_build_time))

            # Search
            total_search_time_for_method = 0
            num_queries_processed = 0
            if not query_smiles_list: print("No queries to process.")

            for q_smiles in query_smiles_list:
                if not q_smiles: continue # Skip if query SMILES is empty/None
                
                results, search_time = searcher.search(q_smiles, n_neighbors=N_NEIGHBORS_TO_FIND)
                total_search_time_for_method += search_time
                num_queries_processed +=1
                
                # print(f"Query: {q_smiles[:30]}... Found {len(results)} neighbors.")
                # for r_id, r_smiles, r_dist in results:
                #     print(f"  ID: {r_id}, SMILES: {r_smiles[:30]}..., Distance: {r_dist:.4f}")

            if num_queries_processed > 0:
                avg_search_time = total_search_time_for_method / num_queries_processed
                benchmark_results["search_times"].append((method_name, avg_search_time))
                print(f"Avg search time for {method_name}: {avg_search_time:.6f}s over {num_queries_processed} queries.")
            else:
                benchmark_results["search_times"].append((method_name, 0)) # No queries processed
                print(f"No queries processed for {method_name}.")


    # --- Report Generation (Data for Plots) ---
    print("\n\n--- BENCHMARKING REPORT DATA ---")
    print(f"Fingerprint Generation Time (Morgan fp, R={searcher.fp_radius}, NBits={searcher.fp_nbits}): {benchmark_results['build_fp_time']:.4f}s for {len(all_smiles)} molecules.")

    print("\nCluster Index Build Times (excluding fingerprint generation):")
    # Filter out BruteForce as it has no cluster build time
    cluster_build_data = [item for item in benchmark_results["build_cluster_time"] if "BruteForce" not in item[0]]
    if not cluster_build_data:
        print("No clustering algorithms were benchmarked for build time.")
    else:
        for name, t_build in cluster_build_data:
            print(f"- {name}: {t_build:.4f}s")
        
        # Data for plot 1: Index Creation Time
        plot1_labels_build = [item[0] for item in cluster_build_data]
        plot1_values_build = [item[1] for item in cluster_build_data]

        fig1, ax1 = plt.subplots()
        ax1.bar(plot1_labels_build, plot1_values_build)
        ax1.set_ylabel('Time (s)')
        ax1.set_xlabel('Clustering Method')
        ax1.set_title('Index Creation Time (Clustering Step)')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("plot_index_creation_time.png") # User can uncomment to save
        print("Plot 1 (Index Creation Time) data generated. Described in report.")


    print("\nAverage Search Times (per query):")
    if not benchmark_results["search_times"]:
        print("No search benchmarks were run.")
    else:
        for name, t_search in benchmark_results["search_times"]:
            print(f"- {name}: {t_search:.6f}s")

        # Data for plot 2: Average Search Time
        plot2_labels_search = [item[0] for item in benchmark_results["search_times"]]
        plot2_values_search = [item[1] for item in benchmark_results["search_times"]]

        fig2, ax2 = plt.subplots()
        ax2.bar(plot2_labels_search, plot2_values_search)
        ax2.set_ylabel('Average Time (s)')
        ax2.set_xlabel('Search Method (Metric + Algorithm)')
        ax2.set_title(f'Average Search Time (N={N_NEIGHBORS_TO_FIND} neighbors)')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("plot_average_search_time.png") # User can uncomment to save
        print("Plot 2 (Average Search Time) data generated. Described in report.")
    
    print("\nTo view plots, uncomment plt.savefig lines and run the script, or use a Jupyter environment to display them directly.")
    # plt.show() # In a script, this might block; better to save files or use in interactive env.
