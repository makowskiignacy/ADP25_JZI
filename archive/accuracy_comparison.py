import random
import time
import statistics
from typing import List, Tuple, Dict, Any
import json
from dataclasses import dataclass

# Import the existing classes
from archive.search_clusters import TanimotoNNChainSearch, HDBSCANSearch, RDKIT_AVAILABLE, HDBSCAN_AVAILABLE

if RDKIT_AVAILABLE:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

@dataclass
class AccuracyMetrics:
    """Data class to store accuracy metrics for a search method."""
    precision_at_5: float
    recall_at_5: float
    average_precision: float
    search_time: float

class PreciseTanimotoSearch:
    """
    Precise brute-force Tanimoto similarity search for reference.
    Computes similarity to all compounds without clustering.
    """
    
    def __init__(self, morgan_radius: int = 2, morgan_nbits: int = 1024):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for PreciseTanimotoSearch but not found.")
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
    
    def similarity(self, compound1_smiles: str, compound2_smiles: str) -> float:
        """Calculate Tanimoto similarity between two SMILES strings."""
        mol1 = Chem.MolFromSmiles(compound1_smiles)
        mol2 = Chem.MolFromSmiles(compound2_smiles)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = GetMorganFingerprintAsBitVect(mol1, self.morgan_radius, nBits=self.morgan_nbits)
        fp2 = GetMorganFingerprintAsBitVect(mol2, self.morgan_radius, nBits=self.morgan_nbits)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def search_most_similar(self, query_smiles: str, compounds: List[str], n: int) -> List[Tuple[str, float]]:
        """
        Find the n most similar compounds to the query using brute-force search.
        
        Args:
            query_smiles: Query compound SMILES
            compounds: List of all compounds to search through
            n: Number of most similar compounds to return
            
        Returns:
            List of (smiles, similarity_score) tuples, sorted by similarity descending
        """
        similarities = []
        
        for compound in compounds:
            if compound == query_smiles:
                continue  # Skip self-comparison
            
            sim_score = self.similarity(query_smiles, compound)
            similarities.append((compound, sim_score))
        
        # Sort by similarity descending and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

def calculate_precision_at_k(predicted: List[str], actual: List[str], k: int) -> float:
    """Calculate precision@k metric."""
    if not predicted or k == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    relevant_in_predicted = len(set(predicted_k) & set(actual))
    return relevant_in_predicted / min(k, len(predicted_k))

def calculate_recall_at_k(predicted: List[str], actual: List[str], k: int) -> float:
    """Calculate recall@k metric."""
    if not actual:
        return 0.0
    
    predicted_k = predicted[:k]
    relevant_in_predicted = len(set(predicted_k) & set(actual))
    return relevant_in_predicted / len(actual)

def calculate_average_precision(predicted: List[str], actual: List[str]) -> float:
    """Calculate Average Precision (AP) metric."""
    if not actual or not predicted:
        return 0.0
    
    actual_set = set(actual)
    precisions = []
    relevant_count = 0
    
    for i, pred in enumerate(predicted):
        if pred in actual_set:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)
    
    return sum(precisions) / len(actual) if precisions else 0.0

def run_single_search_comparison(
    query_smiles: str,
    compounds: List[str],
    precise_searcher: PreciseTanimotoSearch,
    search_methods: Dict[str, Any],
    k: int = 5
) -> Dict[str, AccuracyMetrics]:
    """
    Run a single search comparison between different methods.
    
    Args:
        query_smiles: The query compound
        compounds: List of all compounds
        precise_searcher: Reference precise search method
        search_methods: Dictionary of {method_name: (searcher, clusters)} pairs
        k: Number of top results to compare
        
    Returns:
        Dictionary of method names to their accuracy metrics
    """
    # Get ground truth from precise method
    start_time = time.time()
    ground_truth = precise_searcher.search_most_similar(query_smiles, compounds, k)
    precise_time = time.time() - start_time
    
    ground_truth_smiles = [smiles for smiles, _ in ground_truth]
    
    results = {}
    
    # Add precise method results
    results["Precise"] = AccuracyMetrics(
        precision_at_5=1.0,  # Perfect by definition
        recall_at_5=1.0,     # Perfect by definition
        average_precision=1.0,  # Perfect by definition
        search_time=precise_time
    )
    
    # Test each clustering-based method
    for method_name, (searcher, clusters) in search_methods.items():
        start_time = time.time()
        try:
            predicted_results = searcher.search_most_similar_in_clusters(query_smiles, clusters, k)
            search_time = time.time() - start_time
            
            predicted_smiles = [smiles for smiles, _ in predicted_results]
            
            # Calculate metrics
            precision = calculate_precision_at_k(predicted_smiles, ground_truth_smiles, k)
            recall = calculate_recall_at_k(predicted_smiles, ground_truth_smiles, k)
            avg_precision = calculate_average_precision(predicted_smiles, ground_truth_smiles)
            
            results[method_name] = AccuracyMetrics(
                precision_at_5=precision,
                recall_at_5=recall,
                average_precision=avg_precision,
                search_time=search_time
            )
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[method_name] = AccuracyMetrics(0.0, 0.0, 0.0, float('inf'))
    
    return results

def main():
    """Main function to run the accuracy comparison."""
    if not RDKIT_AVAILABLE:
        print("RDKit is not available. Cannot perform comparison. Exiting.")
        return
    
    # Configuration
    file_path = "chembl_35_chemreps.txt"
    num_compounds_to_load = 1000
    num_searches = 10
    top_k = 5
    
    print(f"Loading {num_compounds_to_load} SMILES from {file_path}...")
    smiles_list = TanimotoNNChainSearch.read_smiles_from_tsv(
        file_path,
        smiles_column_index=1,
        delimiter='\t',
        skip_header=True,
        limit=num_compounds_to_load
    )
    
    if len(smiles_list) < 100:
        print(f"Only {len(smiles_list)} SMILES loaded. Need at least 100 for meaningful comparison.")
        return
    
    print(f"Loaded {len(smiles_list)} SMILES strings.")
    
    # Initialize searchers and perform clustering
    precise_searcher = PreciseTanimotoSearch()
    search_methods = {}
    
    # TanimotoNNChainSearch
    print("Initializing TanimotoNNChainSearch...")
    nn_searcher = TanimotoNNChainSearch(similarity_threshold=0.7)
    start_time = time.time()
    nn_clusters = nn_searcher.cluster(smiles_list)
    nn_cluster_time = time.time() - start_time
    search_methods["TanimotoNNChain"] = (nn_searcher, nn_clusters)
    print(f"TanimotoNNChainSearch clustering completed in {nn_cluster_time:.2f}s, {len(nn_clusters)} clusters")
    
    # HDBSCANSearch (if available)
    if HDBSCAN_AVAILABLE:
        print("Initializing HDBSCANSearch...")
        hdbscan_searcher = HDBSCANSearch(
            min_cluster_size=5,
            min_samples=None,
            metric='jaccard',
            allow_single_cluster=True
        )
        start_time = time.time()
        hdbscan_clusters = hdbscan_searcher.cluster(smiles_list)
        hdbscan_cluster_time = time.time() - start_time
        search_methods["HDBSCAN"] = (hdbscan_searcher, hdbscan_clusters)
        print(f"HDBSCANSearch clustering completed in {hdbscan_cluster_time:.2f}s, {len(hdbscan_clusters)} clusters")
    else:
        print("HDBSCAN not available, skipping...")
    
    # Perform searches
    print(f"\nPerforming {num_searches} search comparisons...")
    
    # Randomly select query compounds
    random.seed(42)  # For reproducibility
    query_compounds = random.sample(smiles_list, num_searches)
    
    all_results = []
    
    for i, query_smiles in enumerate(query_compounds):
        print(f"Search {i+1}/{num_searches}: {query_smiles[:50]}...")
        
        search_results = run_single_search_comparison(
            query_smiles, smiles_list, precise_searcher, search_methods, top_k
        )
        all_results.append(search_results)
    
    # Aggregate results
    print("\n" + "="*80)
    print("ACCURACY COMPARISON RESULTS")
    print("="*80)
    
    method_names = list(all_results[0].keys())
    
    for method_name in method_names:
        print(f"\n{method_name}:")
        print("-" * len(method_name))
        
        precisions = [result[method_name].precision_at_5 for result in all_results]
        recalls = [result[method_name].recall_at_5 for result in all_results]
        avg_precisions = [result[method_name].average_precision for result in all_results]
        search_times = [result[method_name].search_time for result in all_results]
        
        print(f"Precision@{top_k}: {statistics.mean(precisions):.4f} ± {statistics.stdev(precisions) if len(precisions) > 1 else 0:.4f}")
        print(f"Recall@{top_k}:    {statistics.mean(recalls):.4f} ± {statistics.stdev(recalls) if len(recalls) > 1 else 0:.4f}")
        print(f"Avg Precision:   {statistics.mean(avg_precisions):.4f} ± {statistics.stdev(avg_precisions) if len(avg_precisions) > 1 else 0:.4f}")
        print(f"Search Time:     {statistics.mean(search_times):.6f}s ± {statistics.stdev(search_times) if len(search_times) > 1 else 0:.6f}s")
    
    # Save detailed results to JSON
    output_file = "accuracy_comparison_results.json"
    detailed_results = {
        "configuration": {
            "num_compounds": len(smiles_list),
            "num_searches": num_searches,
            "top_k": top_k,
            "query_compounds": query_compounds
        },
        "individual_results": []
    }
    
    for i, (query, results) in enumerate(zip(query_compounds, all_results)):
        search_data = {
            "search_id": i + 1,
            "query_smiles": query,
            "methods": {}
        }
        
        for method_name, metrics in results.items():
            search_data["methods"][method_name] = {
                "precision_at_5": metrics.precision_at_5,
                "recall_at_5": metrics.recall_at_5,
                "average_precision": metrics.average_precision,
                "search_time": metrics.search_time
            }
        
        detailed_results["individual_results"].append(search_data)
    
    # Add summary statistics
    detailed_results["summary"] = {}
    for method_name in method_names:
        precisions = [result[method_name].precision_at_5 for result in all_results]
        recalls = [result[method_name].recall_at_5 for result in all_results]
        avg_precisions = [result[method_name].average_precision for result in all_results]
        search_times = [result[method_name].search_time for result in all_results]
        
        detailed_results["summary"][method_name] = {
            "precision_at_5_mean": statistics.mean(precisions),
            "precision_at_5_std": statistics.stdev(precisions) if len(precisions) > 1 else 0,
            "recall_at_5_mean": statistics.mean(recalls),
            "recall_at_5_std": statistics.stdev(recalls) if len(recalls) > 1 else 0,
            "average_precision_mean": statistics.mean(avg_precisions),
            "average_precision_std": statistics.stdev(avg_precisions) if len(avg_precisions) > 1 else 0,
            "search_time_mean": statistics.mean(search_times),
            "search_time_std": statistics.stdev(search_times) if len(search_times) > 1 else 0
        }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()