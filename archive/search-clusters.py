from abc import ABC, abstractmethod
import time
import json

from abc import ABC, abstractmethod
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    from rdkit import rdBase # Import rdBase
    RDKIT_AVAILABLE = True
    rdBase.DisableLog('rdApp.warning') # Disable RDKit warnings
except ImportError:
    RDKIT_AVAILABLE = False
try:
    import hdbscan
    import numpy as np
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

class CompoundSearch(ABC):
    @abstractmethod
    def similarity(self, compound1: str, compound2: str) -> float:
        pass

    @abstractmethod
    def cluster(self, compounds: list[str]) -> dict[str, dict]:
        pass



class TanimotoNNChainSearch(CompoundSearch):
    """
    A CompoundSearch implementation that uses Tanimoto similarity
    for SMILES strings and a nearest-neighbor chain algorithm for clustering.
    """
    def __init__(self, similarity_threshold: float = 0.7, morgan_radius: int = 2, morgan_nbits: int = 1024):
        """
        Initializes the search class.

        Args:
            similarity_threshold: The minimum similarity for a compound to be added to a chain.
            morgan_radius: Radius for Morgan fingerprints.
            morgan_nbits: Number of bits for Morgan fingerprints.
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for TanimotoNNChainSearch but not found.")
        self.similarity_threshold = similarity_threshold
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits

    def similarity(self, compound1_smiles: str, compound2_smiles: str) -> float:
        """
        Calculates Tanimoto similarity between two SMILES strings.
        """
        mol1 = Chem.MolFromSmiles(compound1_smiles)
        mol2 = Chem.MolFromSmiles(compound2_smiles)

        if mol1 is None or mol2 is None:
            # Invalid SMILES strings cannot be compared
            return 0.0

        fp1 = GetMorganFingerprintAsBitVect(mol1, self.morgan_radius, nBits=self.morgan_nbits)
        fp2 = GetMorganFingerprintAsBitVect(mol2, self.morgan_radius, nBits=self.morgan_nbits)

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def cluster(self, compounds: list[str]) -> dict[str, dict[str, dict]]:
        """
        Clusters compounds using a nearest-neighbor chain algorithm.

        The algorithm works as follows:
        1. Take an unclustered compound to start a new chain (cluster). This compound becomes the representative key for the cluster.
        2. Find the unclustered compound most similar to the *last* compound added to the current chain.
        3. If its similarity is above the threshold, add it to the chain.
        4. Repeat step 2-3 until the chain can no longer be extended.
        5. If unclustered compounds remain, go to step 1 to start a new chain.

        Returns:
            A dictionary where keys are representative SMILES strings (the first compound in each cluster chain)
            and values are dictionaries of compounds (SMILES strings) belonging
            to that cluster, mapping to an empty dictionary.
            Example: {"CCO": {"CCO": {}, "CCN": {}}, "c1ccccc1": {"c1ccccc1": {}}}
        """
        if not compounds:
            return {}

        # Ensure unique compounds to avoid issues with set operations if duplicates exist
        unique_compounds = sorted(list(set(compounds))) # Sort for deterministic behavior if multiple starting points are equivalent
        
        clusters: dict[str, dict[str, dict]] = {} # Keys are representative SMILES
        remaining_compounds = set(unique_compounds)

        processed_in_iteration = True # To handle cases where no compound can start a chain

        while remaining_compounds and processed_in_iteration:
            processed_in_iteration = False # Reset for current pass over remaining_compounds
            
            potential_starters = list(remaining_compounds) 
            
            for seed_compound in potential_starters:
                if seed_compound not in remaining_compounds: # Already processed in this outer loop iteration
                    continue

                current_chain_compound = seed_compound
                
                # Start a new chain, with seed_compound as the representative
                # current_cluster_compounds will hold all members of this cluster
                current_cluster_compounds: dict[str, dict] = {current_chain_compound: {}}
                remaining_compounds.remove(current_chain_compound)
                processed_in_iteration = True

                chain_extended = True
                while chain_extended:
                    chain_extended = False
                    best_next_compound = None
                    max_similarity = -1.0 # Must be at least threshold

                    for candidate_compound in list(remaining_compounds):
                        sim = self.similarity(current_chain_compound, candidate_compound)
                        if sim >= self.similarity_threshold and sim > max_similarity:
                            max_similarity = sim
                            best_next_compound = candidate_compound
                    
                    if best_next_compound is not None:
                        current_cluster_compounds[best_next_compound] = {}
                        remaining_compounds.remove(best_next_compound)
                        current_chain_compound = best_next_compound # Move to the new end of the chain
                        chain_extended = True
                
                if current_cluster_compounds: # If the chain (even if single item) was formed
                    # The seed_compound is the key representing this cluster
                    clusters[seed_compound] = current_cluster_compounds
                # If seed_compound couldn't start a chain that attracted others, it's now processed.
                # If it was the only one in its cluster, that's fine.

        # Any compounds left in remaining_compounds could not form or join a chain
        # based on the threshold. Add them as singleton clusters,
        # where the compound itself is the representative key.
        while remaining_compounds:
            singleton_compound = remaining_compounds.pop()
            clusters[singleton_compound] = {singleton_compound: {}}
            
        return clusters

    @staticmethod
    def read_smiles_from_file(file_path: str) -> list[str]:
        """
        Reads SMILES strings from a file.
        Each line in the file should represent one SMILES string.

        Args:
            file_path: The path to the file containing SMILES strings.

        Returns:
            A list of SMILES strings.
        """
        smiles_list = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    smiles = line.strip()
                    if smiles: # Ensure non-empty lines
                        smiles_list.append(smiles)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return []
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")
            return []
        return smiles_list

    def search_most_similar_in_clusters(self, query_smiles: str, clusters: dict[str, dict[str, dict]], n: int) -> list[tuple[str, float]]:
        """
        Searches for the n most similar compounds to a query SMILES string
        within the provided clusters.

        Args:
            query_smiles: The SMILES string of the query compound.
            clusters: A dictionary of clusters, as returned by the cluster() method.
                      Example: {"cluster_0": {"CCO": {}, "CCN": {}}, ...}
            n: The number of most similar compounds to return.

        Returns:
            A list of tuples, where each tuple contains (compound_smiles, similarity_score),
            sorted by similarity in descending order. Returns up to n compounds.
            The query compound itself is excluded from the results.
        """
        if not RDKIT_AVAILABLE:
            print("Warning: RDKit not available. Cannot perform similarity search.")
            return []
        if n <= 0:
            return []

        all_similarities: list[tuple[str, float]] = []

        for _cluster_name, compounds_in_cluster in clusters.items():
            for compound_smiles in compounds_in_cluster.keys():
                if compound_smiles == query_smiles:
                    continue # Skip comparing the query to itself

                similarity_score = self.similarity(query_smiles, compound_smiles)
                all_similarities.append((compound_smiles, similarity_score))

        # Sort by similarity score in descending order
        all_similarities.sort(key=lambda x: x[1], reverse=True)

        return all_similarities[:n]
    
    @staticmethod
    def read_smiles_from_tsv(file_path: str, smiles_column_index: int = 1, delimiter: str = '\t', skip_header: bool = True, limit: int | None = None) -> list[str]:
        """
        Reads SMILES strings from a delimited file (e.g., TSV, CSV).

        Args:
            file_path: The path to the file.
            smiles_column_index: The 0-based index of the column containing SMILES strings.
            delimiter: The delimiter used in the file.
            skip_header: Whether to skip the first line (header).
            limit: Maximum number of SMILES strings to read. None for no limit.

        Returns:
            A list of SMILES strings.
        """
        smiles_list = []
        try:
            with open(file_path, 'r') as f:
                if skip_header:
                    next(f, None)  # Skip the header line
                
                for _ in range(limit if limit is not None else float('inf')): # type: ignore
                    line = f.readline()
                    if not line: # End of file
                        break
                    parts = line.strip().split(delimiter)
                    if len(parts) > smiles_column_index:
                        smiles = parts[smiles_column_index]
                        if smiles:  # Ensure non-empty SMILES
                            smiles_list.append(smiles)
                    if limit is not None and len(smiles_list) >= limit:
                        break
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return []
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")
            return []
        return smiles_list
    
class HDBSCANSearch(CompoundSearch):
    """
    A CompoundSearch implementation that uses Tanimoto similarity
    for SMILES strings and HDBSCAN for clustering.
    """
    def __init__(self, morgan_radius: int = 2, morgan_nbits: int = 1024,
                 min_cluster_size: int = 5, min_samples: int | None = None,
                 cluster_selection_epsilon: float = 0.0, metric: str = 'jaccard',
                 allow_single_cluster: bool = False, **hdbscan_kwargs):
        """
        Initializes the HDBSCAN search class.

        Args:
            morgan_radius: Radius for Morgan fingerprints.
            morgan_nbits: Number of bits for Morgan fingerprints.
            min_cluster_size: The minimum size of clusters for HDBSCAN.
            min_samples: The number of samples in a neighborhood for a point to be
                         considered as a core point by HDBSCAN. If None, defaults to min_cluster_size.
            metric: The metric to use for HDBSCAN (e.g., 'jaccard' for bit vectors).
            cluster_selection_epsilon: Epsilon value for HDBSCAN's flat cluster extraction.
            allow_single_cluster: Whether HDBSCAN can identify a single cluster.
            **hdbscan_kwargs: Additional keyword arguments for hdbscan.HDBSCAN.
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for HDBSCANSearch but not found.")
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan library is required for HDBSCANSearch but not found.")
        
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.allow_single_cluster = allow_single_cluster
        self.hdbscan_kwargs = hdbscan_kwargs

    def similarity(self, compound1_smiles: str, compound2_smiles: str) -> float:
        """
        Calculates Tanimoto similarity between two SMILES strings.
        """
        mol1 = Chem.MolFromSmiles(compound1_smiles)
        mol2 = Chem.MolFromSmiles(compound2_smiles)

        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = GetMorganFingerprintAsBitVect(mol1, self.morgan_radius, nBits=self.morgan_nbits)
        fp2 = GetMorganFingerprintAsBitVect(mol2, self.morgan_radius, nBits=self.morgan_nbits)

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def cluster(self, compounds: list[str]) -> dict[str, dict[str, dict]]:
        """
        Clusters compounds using HDBSCAN on their Morgan fingerprints.

        Returns:
            A dictionary where keys are representative SMILES strings
            (first compound in each HDBSCAN cluster or the compound itself for noise points)
            and values are dictionaries of compounds (SMILES strings) belonging
            to that cluster, mapping to an empty dictionary.
            Example: {"CCO": {"CCO": {}, "CCN": {}}, "c1ccccc1": {"c1ccccc1": {}}}
        """
        if not compounds:
            return {}

        unique_compounds = sorted(list(set(compounds))) # Process unique compounds

        valid_smiles_list = []
        fingerprints_list = []

        for smiles in unique_compounds:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_nbits)
                np_fp = np.zeros((self.morgan_nbits,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(fp, np_fp)
                fingerprints_list.append(np_fp)
                valid_smiles_list.append(smiles)
            # Invalid SMILES will be handled later as singletons if not clustered

        if not fingerprints_list: # No valid SMILES found
            result_clusters = {}
            for s in unique_compounds: # All original unique compounds as singletons
                result_clusters[s] = {s: {}}
            return result_clusters

        fp_matrix = np.array(fingerprints_list, dtype=np.int32)
        
        # Ensure enough samples for HDBSCAN to potentially form clusters
        # HDBSCAN itself will label points as noise if clusters can't be formed based on parameters
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                    min_samples=self.min_samples,
                                    metric=self.metric,
                                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                                    allow_single_cluster=self.allow_single_cluster,
                                    **self.hdbscan_kwargs)
        try:
            cluster_labels = clusterer.fit_predict(fp_matrix)
        except Exception as e:
            print(f"Error during HDBSCAN clustering: {e}. Treating all compounds as singletons.")
            result_clusters = {}
            for s in unique_compounds:
                result_clusters[s] = {s: {}}
            return result_clusters

        # Group SMILES by HDBSCAN cluster label
        # valid_smiles_list[i] corresponds to cluster_labels[i]
        labeled_smiles_map: dict[int, list[str]] = {}
        for i, label in enumerate(cluster_labels):
            smiles = valid_smiles_list[i]
            if label not in labeled_smiles_map:
                labeled_smiles_map[label] = []
            labeled_smiles_map[label].append(smiles)

        result_clusters: dict[str, dict[str, dict]] = {}
        processed_smiles_set = set()

        # Process actual clusters (label >= 0)
        for label_id, members in labeled_smiles_map.items():
            if label_id == -1: # Noise points are handled separately
                continue
            if not members:
                continue
            
            # Use the first SMILES (sorted for determinism) as the representative for this HDBSCAN cluster
            representative_smiles = sorted(members)[0] 
            
            current_cluster_members_dict: dict[str, dict] = {}
            for member_smiles in members:
                current_cluster_members_dict[member_smiles] = {}
                processed_smiles_set.add(member_smiles)
            result_clusters[representative_smiles] = current_cluster_members_dict

        # Handle noise points (label == -1) from HDBSCAN as singleton clusters
        if -1 in labeled_smiles_map:
            for smiles in labeled_smiles_map[-1]:
                if smiles not in processed_smiles_set: # Should be true for noise points
                    result_clusters[smiles] = {smiles: {}}
                    processed_smiles_set.add(smiles)
        
        # Add any original unique compounds that were not processed (e.g., invalid SMILES) as singletons
        for smiles_item in unique_compounds:
            if smiles_item not in processed_smiles_set:
                result_clusters[smiles_item] = {smiles_item: {}}
                
        return result_clusters

    def search_most_similar_in_clusters(self, query_smiles: str, clusters: dict[str, dict[str, dict]], n: int) -> list[tuple[str, float]]:
        """
        Searches for the n most similar compounds to a query SMILES string
        within the provided clusters. (Identical to TanimotoNNChainSearch's method)
        """
        if n <= 0:
            return []

        all_similarities: list[tuple[str, float]] = []

        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            print(f"Warning: Query SMILES '{query_smiles}' is invalid. Cannot perform search.")
            return []
        
        fp_query = GetMorganFingerprintAsBitVect(query_mol, self.morgan_radius, nBits=self.morgan_nbits)

        for _cluster_representative, compounds_in_cluster in clusters.items():
            for compound_smiles in compounds_in_cluster.keys():
                if compound_smiles == query_smiles:
                    continue 

                mol_db = Chem.MolFromSmiles(compound_smiles)
                if mol_db is None: # Skip invalid SMILES in database
                    continue
                
                fp_db = GetMorganFingerprintAsBitVect(mol_db, self.morgan_radius, nBits=self.morgan_nbits)
                similarity_score = DataStructs.TanimotoSimilarity(fp_query, fp_db)
                all_similarities.append((compound_smiles, similarity_score))

        all_similarities.sort(key=lambda x: x[1], reverse=True)
        return all_similarities[:n]


def similarity1(compound1, compound2):
    return ord(compound1)


#creates a lists of keys and clasters for sorting by key
def dict_to_list(dict):
    keys = dict.keys()
    dict_list = []
    for key in keys:
        dict_list.append({'key' : key, 'claster': dict[key]})
    return dict_list
    


#gets n most similar compunds to given compunds with given similarity function
def get_similar_compounds(compound, claster, similarity_func, n):
    if False == bool(claster):
        return []
    claster_list = dict_to_list(claster)
    def sorting_func(a):
        return similarity_func(a['key'], compound)
    claster_list.sort(key = sorting_func)
    n_remain = n
    compound_list = []
    for i in claster_list:
        new_compounds = get_similar_compounds(compound, i['claster'], similarity_func, n_remain)
        if len(new_compounds) == 0:
            new_compounds = i['key']
        n_remain -= len(new_compounds)
        compound_list += new_compounds
        if n_remain<=0:
            return compound_list

    return compound_list

def main():
    """
    Main function to demonstrate clustering and searching.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit is not available. Cannot perform clustering and search. Exiting.")
        return

    file_path = "chembl_35_chemreps.txt"  # Ensure this file exists or change path
    num_compounds_to_load = 20000
    # A moderately complex query SMILES
    query_smiles = "O=C(Nc1ccc(F)cc1)c1nnc(SCC(=O)Nc2ccccc2)o1" 
    # Example query: "CC(C)C[C@@H](C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CS)C(=O)O)N(C)C(=O)CN(C)C(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccsc1)NC(=O)CNC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](N)CCCNC(=N)N)C(C)(C)S)[C@@H](C)O"
    num_similar_to_find = 5

    print(f"Loading first {num_compounds_to_load} SMILES from {file_path}...")
    smiles_list = TanimotoNNChainSearch.read_smiles_from_tsv(
        file_path, 
        smiles_column_index=1, 
        delimiter='\t', 
        skip_header=True, 
        limit=num_compounds_to_load
    )

    if not smiles_list:
        print("No SMILES strings loaded. Exiting.")
        return
    
    print(f"Loaded {len(smiles_list)} SMILES strings.")

    # # --- Original TanimotoNNChainSearch Analysis ---
    # print("\n--- Running TanimotoNNChainSearch ---")
    # searcher_nn = TanimotoNNChainSearch(similarity_threshold=0.7) 
    
    # print("Clustering compounds with TanimotoNNChainSearch...")
    # start_clustering_time_nn = time.time()
    # clusters_nn = searcher_nn.cluster(smiles_list)
    # end_clustering_time_nn = time.time()
    # print(f"TanimotoNNChainSearch clustering completed in {end_clustering_time_nn - start_clustering_time_nn:.2f} seconds.")
    # print(f"Found {len(clusters_nn)} clusters with TanimotoNNChainSearch.")

    # if clusters_nn:
    #     output_json_path_nn = "clusters_output_nn.json"
    #     try:
    #         with open(output_json_path_nn, 'w') as f_json:
    #             json.dump(clusters_nn, f_json, indent=4)
    #         print(f"TanimotoNNChainSearch clusters saved to {output_json_path_nn}")
    #     except IOError as e:
    #         print(f"Error saving TanimotoNNChainSearch clusters to JSON: {e}")

    #     print(f"\nSearching with TanimotoNNChainSearch for {num_similar_to_find} most similar compounds to: {query_smiles[:50]}...")
    #     start_search_time_nn = time.time()
    #     similar_compounds_nn = searcher_nn.search_most_similar_in_clusters(query_smiles, clusters_nn, num_similar_to_find)
    #     end_search_time_nn = time.time()
    #     search_duration_nn = end_search_time_nn - start_search_time_nn
    #     print(f"TanimotoNNChainSearch search completed in {search_duration_nn:.4f} seconds.")

    #     if similar_compounds_nn:
    #         print(f"\nTop {len(similar_compounds_nn)} similar compounds found by TanimotoNNChainSearch:")
    #         for i, (smiles, score) in enumerate(similar_compounds_nn):
    #             print(f"{i+1}. SMILES: {smiles[:70]}..., Similarity: {score:.4f}") # Truncate long SMILES
    #     else:
    #         print("No similar compounds found by TanimotoNNChainSearch.")
    # else:
    #     print("No clusters were formed by TanimotoNNChainSearch. Cannot perform search.")

    # --- HDBSCANSearch Analysis ---
    if not HDBSCAN_AVAILABLE:
        print("\nHDBSCAN library not found. Skipping HDBSCANSearch analysis.")
    else:
        print("\n--- Running HDBSCANSearch ---")
        # Adjust HDBSCAN parameters as needed for your dataset
        searcher_hdbscan = HDBSCANSearch(
            morgan_radius=2, 
            morgan_nbits=1024,
            min_cluster_size=5,    # Minimum number of compounds to form a cluster
            min_samples=None,      # Defaults to min_cluster_size. Affects how conservative clustering is.
            metric='jaccard',      # Jaccard distance (1 - Tanimoto) for binary fingerprints
            allow_single_cluster=True # Useful if the dataset might form one large cluster
        )
        
        print("Clustering compounds with HDBSCANSearch...")
        start_clustering_time_hdbscan = time.time()
        clusters_hdbscan = searcher_hdbscan.cluster(smiles_list)
        end_clustering_time_hdbscan = time.time()
        print(f"HDBSCANSearch clustering completed in {end_clustering_time_hdbscan - start_clustering_time_hdbscan:.2f} seconds.")
        # Note: The number of "clusters" includes singleton noise points.
        num_actual_hdbscan_clusters = sum(1 for rep, members in clusters_hdbscan.items() if len(members) > 1 or rep not in members or len(members[rep]) >0 ) # Heuristic
        
        hdbscan_cluster_count = 0
        noise_points_count = 0
        for rep, members in clusters_hdbscan.items():
            # A simple way to distinguish noise (singletons where rep is the only member)
            # from actual clusters. This depends on how representatives are chosen for noise.
            # My implementation makes noise points their own representative.
            if len(members) == 1 and rep in members:
                 noise_points_count +=1
            else:
                 hdbscan_cluster_count +=1
        print(f"Found {hdbscan_cluster_count} HDBSCAN clusters and {noise_points_count} noise points (treated as singletons). Total entries: {len(clusters_hdbscan)}")


        if not clusters_hdbscan:
            print("No clusters/noise points were identified by HDBSCANSearch.")
        else:
            output_json_path_hdbscan = "clusters_output_hdbscan.json"
            try:
                with open(output_json_path_hdbscan, 'w') as f_json:
                    json.dump(clusters_hdbscan, f_json, indent=4)
                print(f"HDBSCANSearch clusters saved to {output_json_path_hdbscan}")
            except IOError as e:
                print(f"Error saving HDBSCANSearch clusters to JSON: {e}")

            print(f"\nSearching with HDBSCANSearch for {num_similar_to_find} most similar compounds to: {query_smiles[:50]}...")
            start_search_time_hdbscan = time.time()
            similar_compounds_hdbscan = searcher_hdbscan.search_most_similar_in_clusters(query_smiles, clusters_hdbscan, num_similar_to_find)
            end_search_time_hdbscan = time.time()
            search_duration_hdbscan = end_search_time_hdbscan - start_search_time_hdbscan
            print(f"HDBSCANSearch search completed in {search_duration_hdbscan:.4f} seconds.")

            if similar_compounds_hdbscan:
                print(f"\nTop {len(similar_compounds_hdbscan)} similar compounds found by HDBSCANSearch:")
                for i, (smiles, score) in enumerate(similar_compounds_hdbscan):
                    print(f"{i+1}. SMILES: {smiles[:70]}..., Similarity: {score:.4f}") # Truncate long SMILES
            else:
                print("No similar compounds found by HDBSCANSearch.")

if __name__ == "__main__":
    main()
