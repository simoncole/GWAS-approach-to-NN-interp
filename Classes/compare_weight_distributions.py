import numpy as np

class CompareWeightDistributions:
    def __init__(self, reference_WB_object, phenotype_WB_object, kl_threshold=0.01, chi2_alpha=0.05):
        """
        Parameters:
        - reference_WB_object, phenotype_WB_object: Two instances of WeightBinning, each having already loaded,
          binned, normalized, and fitted the distributions.
        - kl_threshold: KL divergence threshold for deciding if a pair of distributions is considered a "match".
        - chi2_alpha: Significance level for the chi-square test (used later for bin-wise comparisons).
        """
        self.reference_WB_object = reference_WB_object
        self.phenotype_WB_object = phenotype_WB_object
        self.kl_threshold = kl_threshold
        self.chi2_alpha = chi2_alpha
        self.layer_weight_distributions_counts_ref = reference_WB_object.layer_weight_distributions_counts
        self.layer_weight_distributions_counts_phen = phenotype_WB_object.layer_weight_distributions_counts
        assert hasattr(self.reference_WB_object, 'gmm_models') and hasattr(self.phenotype_WB_object, 'gmm_models'), "Fit GMM models first."

    def _compare_cluster_distributions(self, ref_clusters, phen_clusters, ref_unique_clusters, phen_unique_clusters, ref_models, phen_models):
        """
        Compare each unique reference cluster with each unique phenotype cluster.
        
        Parameters:
        - ref_clusters: List of unique reference cluster IDs
        - phen_clusters: List of unique phenotype cluster IDs 
        - ref_unique_clusters: Dictionary mapping reference cluster IDs to positions
        - phen_unique_clusters: Dictionary mapping phenotype cluster IDs to positions
        - ref_models: Reference GMM models
        - phen_models: Phenotype GMM models
        
        Returns:
        - comparison_matrix: Matrix of KL divergences between clusters
        - matches: List of matching cluster pairs (ref_id, phen_id, kl_divergence)
        """
        # Create matrix for storing KL divergences between unique clusters
        comparison_matrix = np.full((len(ref_clusters), len(phen_clusters)), np.inf)
        
        # Compare each unique reference cluster with each unique phenotype cluster
        matches = []
        for i, ref_cluster_id in enumerate(ref_clusters):
            ref_pos = ref_unique_clusters[ref_cluster_id]
            ref_model = ref_models[ref_pos[0], ref_pos[1]]
            
            for j, phen_cluster_id in enumerate(phen_clusters):
                phen_pos = phen_unique_clusters[phen_cluster_id]
                phen_model = phen_models[phen_pos[0], phen_pos[1]]
                
                assert ref_model is not None and phen_model is not None, f"Models are None for ref_cluster_id: {ref_cluster_id}, phen_cluster_id: {phen_cluster_id}"
                
                # Compute symmetrized KL divergence
                kl_forward = self.reference_WB_object.compute_kl_divergence(
                    ref_model, phen_model, multi_peak=True)
                kl_backward = self.reference_WB_object.compute_kl_divergence(
                    phen_model, ref_model, multi_peak=True)
                kl_divergence = (kl_forward + kl_backward) / 2
                
                # Store KL divergence in the comparison matrix
                comparison_matrix[i, j] = kl_divergence
                
                # Check if the distributions match based on KL threshold
                if kl_divergence < self.kl_threshold:
                    matches.append((ref_cluster_id, phen_cluster_id, kl_divergence))
                    
        return comparison_matrix, matches

    def _create_position_mappings(self, ref_cluster_indices, phen_cluster_indices, ref_clusters, phen_clusters, comparison_matrix):
        """
        Create a detailed mapping of positions to their cluster information and KL divergence.
        
        Parameters:
        - ref_cluster_indices: 2D array of reference cluster indices
        - phen_cluster_indices: 2D array of phenotype cluster indices
        - ref_clusters: List of unique reference cluster IDs
        - phen_clusters: List of unique phenotype cluster IDs
        - comparison_matrix: KL divergence matrix between clusters
        
        Returns:
        - position_mappings: Dictionary mapping (neuron_idx, from_weight_idx) to cluster info
        - ref_cluster_to_idx: Dictionary mapping reference cluster IDs to matrix indices
        - phen_cluster_to_idx: Dictionary mapping phenotype cluster IDs to matrix indices
        """
        # Create a mapping from cluster indices to their positions in the comparison_matrix
        ref_cluster_to_idx = {cluster_id: i for i, cluster_id in enumerate(ref_clusters)}
        phen_cluster_to_idx = {cluster_id: j for j, cluster_id in enumerate(phen_clusters)}
        
        # Create a detailed mapping of positions and their related clusters
        position_mappings = {}
        for neuron_idx in range(ref_cluster_indices.shape[0]):
            for from_weight_idx in range(ref_cluster_indices.shape[1]):
                ref_cluster_id = ref_cluster_indices[neuron_idx, from_weight_idx]
                phen_cluster_id = phen_cluster_indices[neuron_idx, from_weight_idx]
                
                # Skip positions without valid clusters
                if ref_cluster_id < 0 or phen_cluster_id < 0:
                    continue
                
                i = ref_cluster_to_idx.get(ref_cluster_id)
                j = phen_cluster_to_idx.get(phen_cluster_id)
                
                # Skip if either cluster ID is not in our mapping
                if i is None or j is None:
                    continue
                
                kl_value = comparison_matrix[i, j]
                is_match = kl_value < self.kl_threshold
                
                position_mappings[(neuron_idx, from_weight_idx)] = {
                    'ref_cluster_id': ref_cluster_id,
                    'phen_cluster_id': phen_cluster_id,
                    'kl_divergence': kl_value,
                    'is_match': is_match
                }
                
        return position_mappings, ref_cluster_to_idx, phen_cluster_to_idx

    def compare_unique_distributions(self, layer):
        """
        For a given layer index, compare all unique fitted GMM models from the reference and phenotype WeightBinning objects.
        This function compares each unique distribution in the reference model to every unique distribution in the phenotype model.
        
        Returns:
        - results: A dictionary containing:
            - 'comparison_matrix': 2D matrix of KL divergences between unique distributions
            - 'ref_unique_clusters': Dict mapping reference cluster IDs to representative positions
            - 'phen_unique_clusters': Dict mapping phenotype cluster IDs to representative positions
            - 'matches': List of tuples (ref_cluster_id, phen_cluster_id, kl_divergence) for matched clusters
        """
        # Verify that both objects have cluster indices for the specified layer
        assert hasattr(self.reference_WB_object, 'cluster_indices') and hasattr(self.phenotype_WB_object, 'cluster_indices'), \
            "Cluster indices not available. Run cluster_distributions() first."
        
        # Get cluster indices for the specified layer
        ref_cluster_indices = self.reference_WB_object.cluster_indices[layer]
        phen_cluster_indices = self.phenotype_WB_object.cluster_indices[layer]
        
        # Get GMM models for the specified layer
        ref_models = self.reference_WB_object.gmm_models[layer]
        phen_models = self.phenotype_WB_object.gmm_models[layer]
        
        # Extract unique clusters from both models
        ref_unique_clusters = self._extract_unique_clusters(ref_cluster_indices)
        phen_unique_clusters = self._extract_unique_clusters(phen_cluster_indices)
        
        # Create sorted lists of cluster IDs for consistent indexing
        ref_clusters = sorted(ref_unique_clusters.keys())
        phen_clusters = sorted(phen_unique_clusters.keys())
        
        # Compare distributions between unique clusters
        comparison_matrix, matches = self._compare_cluster_distributions(
            ref_clusters, 
            phen_clusters, 
            ref_unique_clusters, 
            phen_unique_clusters, 
            ref_models, 
            phen_models
        )
        
        # Create position mappings and cluster index mappings
        position_mappings, ref_cluster_to_idx, phen_cluster_to_idx = self._create_position_mappings(
            ref_cluster_indices,
            phen_cluster_indices,
            ref_clusters,
            phen_clusters,
            comparison_matrix
        )
        
        # Construct and return the results
        results = {
            'comparison_matrix': comparison_matrix,
            'ref_unique_clusters': ref_unique_clusters,
            'phen_unique_clusters': phen_unique_clusters,
            'ref_cluster_positions': ref_cluster_to_idx,
            'phen_cluster_positions': phen_cluster_to_idx,
            'matches': matches,
            'position_mappings': position_mappings
        }
        
        return results
        
    def find_matching_unique_distributions(self, layer):
        """
        For a given layer index, compare the fitted models from the reference and phenotype WeightBinning objects.
        Identify "matching" distributions through comparison of the unique cluster indices.
        
        Returns:
        - results: A dictionary with keys as (neuron_idx, from_weight_idx) and values as the computed KL divergence.
                   Additionally, mark those positions as "matched" if KL divergence < self.kl_threshold.
        """
        # Verify that both objects have cluster indices for the specified layer
        assert hasattr(self.reference_WB_object, 'cluster_indices') and hasattr(self.phenotype_WB_object, 'cluster_indices'), \
            "Cluster indices not available. Run cluster_distributions() first."
        
        # Get the GMM models for the specified layer from both objects
        ref_models = self.reference_WB_object.gmm_models[layer]
        phen_models = self.phenotype_WB_object.gmm_models[layer]
        
        # Get the shape of the layer (number of neurons and weights)
        num_neurons, num_from_weights = ref_models.shape
        
        # Initialize results dictionary
        results = {}
        
        # Compare distributions for each weight position
        for neuron_idx in range(num_neurons):
            for from_weight_idx in range(num_from_weights):
                # Get the models at this position
                ref_model = ref_models[neuron_idx, from_weight_idx]
                phen_model = phen_models[neuron_idx, from_weight_idx]
                
                assert ref_model is not None and phen_model is not None, f"Models are None for neuron_idx: {neuron_idx}, from_weight_idx: {from_weight_idx}"
                
                # Compute KL divergence between the distributions
                kl_forward = self.reference_WB_object.compute_kl_divergence(
                    ref_model, phen_model, multi_peak=True)
                kl_backward = self.reference_WB_object.compute_kl_divergence(
                    phen_model, ref_model, multi_peak=True)
                
                # Use symmetrized KL divergence (average of both directions)
                kl_divergence = (kl_forward + kl_backward) / 2
                
                # Determine if the distributions match based on KL threshold
                is_match = kl_divergence < self.kl_threshold
                
                # Store results in the dictionary
                results[(neuron_idx, from_weight_idx)] = {
                    'kl_divergence': kl_divergence,
                    'is_match': is_match
                }
        
        return results
    
    
    def _extract_unique_clusters(self, cluster_indices):
        """
        Extract unique clusters from a cluster indices matrix.
        
        Parameters:
        - cluster_indices: 2D array of cluster indices
        
        Returns:
        - unique_clusters: Dict mapping cluster_id -> (neuron_idx, from_weight_idx)
        """
        unique_clusters = {}
        for neuron_idx in range(cluster_indices.shape[0]):
            for from_weight_idx in range(cluster_indices.shape[1]):
                cluster_id = cluster_indices[neuron_idx, from_weight_idx]
                if cluster_id not in unique_clusters:
                    # Store the first position we find for this cluster
                    unique_clusters[cluster_id] = (neuron_idx, from_weight_idx)
        return unique_clusters
