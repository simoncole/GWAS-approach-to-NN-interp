import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.stats import norm
from scipy.special import expit  # Logistic function
import statsmodels.api as sm
import os
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches


class CompareWeightDistributions:
    def __init__(self, reference_WB_object, phenotype_WB_object, kl_match_threshold=0.01, chi2_alpha=0.05, use_normalized=False):
        """
        Parameters:
        - reference_WB_object, phenotype_WB_object: Two instances of WeightBinning, each having already loaded,
          binned, and fitted the distributions.
        - kl_match_threshold: KL divergence threshold for deciding if a pair of distributions is considered a "match".
        - chi2_alpha: Significance level for the chi-square test (used later for bin-wise comparisons).
        - use_normalized: Boolean flag; if True, use normalized counts for chi-square test.
        """
        self.reference_WB_object = reference_WB_object
        self.phenotype_WB_object = phenotype_WB_object
        self.kl_match_threshold = kl_match_threshold
        self.chi2_alpha = chi2_alpha
        self.use_normalized = use_normalized
        
        # Get the appropriate distributions based on use_normalized flag
        if self.use_normalized:
            # Ensure normalized distributions are available
            if not hasattr(reference_WB_object, 'normalized_distributions'):
                reference_WB_object.normalize_distributions()
            if not hasattr(phenotype_WB_object, 'normalized_distributions'):
                phenotype_WB_object.normalize_distributions()
                
            self.layer_weight_distributions_ref = reference_WB_object.normalized_distributions
            self.layer_weight_distributions_phen = phenotype_WB_object.normalized_distributions
            print("Using normalized distributions for comparison")
        else:
            # Use raw counts
            self.layer_weight_distributions_ref = reference_WB_object.layer_weight_distributions_counts
            self.layer_weight_distributions_phen = phenotype_WB_object.layer_weight_distributions_counts
            print("Using raw count distributions for comparison")
        
        assert hasattr(self.reference_WB_object, 'gmm_models') and hasattr(self.phenotype_WB_object, 'gmm_models'), "Fit GMM models first."
    
    def _merge_tail_bins_pair(self, counts_ref, counts_phen, threshold=0.001):
        """
        Given two 1D arrays (counts_ref and counts_phen) for a single weight position,
        merge adjacent bins at the lower and upper tails if their combined counts are less than
        the threshold fraction (e.g., 0.1% of the total data).
        
        Parameters:
          counts_ref: 1D numpy array of counts from the reference distribution.
          counts_phen: 1D numpy array of counts from the phenotype distribution.
          threshold: Fraction threshold (default 0.001 for 0.1%).
          
        Returns:
          new_counts_ref, new_counts_phen: The new count arrays with tail bins merged.
          merge_info: Tuple (k, m) where k is the number of bins merged at the lower tail and m at the upper tail.
          
        """
        # Filter out None values first
        valid_ref = [c for c in counts_ref if c is not None]
        valid_phen = [c for c in counts_phen if c is not None]
        
        if not valid_ref or not valid_phen:
            # Return empty arrays if one or both distributions have no valid values
            return np.array([]), np.array([]), (0, 0)
        
        combined = np.array(valid_ref) + np.array(valid_phen)
        total = combined.sum()
        if total == 0:
            return np.array([]), np.array([]), (0, 0)
        
        # Determine how many bins at the lower tail to merge:
        cum_lower = np.cumsum(combined)
        k = 0
        while k < len(combined) and cum_lower[k] < threshold * total:
            k += 1
        # If k==0 then no bins fall under threshold; if k==1, then only the first bin is below threshold.
        # We merge if more than one bin collectively is below threshold.
        
        # Determine how many bins at the upper tail to merge:
        rev = combined[::-1]
        cum_upper = np.cumsum(rev)
        m = 0
        while m < len(combined) and cum_upper[m] < threshold * total:
            m += 1
        
        new_counts_ref = []
        new_counts_phen = []
        
        # Merge lower tail if more than one bin qualifies:
        if k > 1:
            new_counts_ref.append(sum(valid_ref[:k]))
            new_counts_phen.append(sum(valid_phen[:k]))
        elif k == 1:
            new_counts_ref.append(valid_ref[0])
            new_counts_phen.append(valid_phen[0])
        else:
            # k == 0: nothing to merge
            pass
        
        # Middle bins: from k to len(combined)-m
        if k < len(combined) - m:
            middle_ref = valid_ref[k:len(combined)-m]
            middle_phen = valid_phen[k:len(combined)-m]
            new_counts_ref.extend(middle_ref)
            new_counts_phen.extend(middle_phen)
        
        # Merge upper tail if more than one bin qualifies:
        if m > 1:
            new_counts_ref.append(sum(valid_ref[-m:]))
            new_counts_phen.append(sum(valid_phen[-m:]))
        elif m == 1:
            new_counts_ref.append(valid_ref[-1])
            new_counts_phen.append(valid_phen[-1])
        
        return np.array(new_counts_ref), np.array(new_counts_phen), (k, m)
    
    def _chi_square_test_for_position(self, counts_ref, counts_phen):
        """
        Helper function to run a chi-square test on a single weight position.
        
        Parameters:
        - counts_ref: 1D numpy array of counts for a given weight position from the reference object.
        - counts_phen: 1D numpy array of counts for the same weight position from the phenotype object.
        
        Returns:
        - chi2_stat: Chi-square statistic.
        - p_value: p-value from the test.
        - residuals: Standardized residuals for each bin.
        - expected: Expected counts computed by the test.
        """
        # Check if either distribution contains only None values (all-zero weights)
        if all(c is None for c in counts_ref) or all(c is None for c in counts_phen):
            return None, None, None, None, None
        
        # Merge the tail bins from both groups so that we avoid very low expected counts.
        new_counts_ref, new_counts_phen, merge_info = self._merge_tail_bins_pair(counts_ref, counts_phen, threshold=0.001)
        
        # If we have no valid bins after merging, return None
        if len(new_counts_ref) == 0 or len(new_counts_phen) == 0:
            return None, None, None, None, None
        
        # Build a contingency table of shape (2, new_number_of_bins)
        contingency_table = np.array([new_counts_ref, new_counts_phen])
        
        # Skip if any row or column sums to zero
        if np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
            return None, None, None, None, None
            
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        residuals = (contingency_table - expected) / np.sqrt(expected)
        return chi2_stat, p_value, residuals, expected, merge_info

    def binwise_chi_square_comparison(self, layer, positions=None, apply_correction=False, correction_method='bonferroni'):
        """
        Perform a bin-wise chi-square comparison for weight positions in a given layer.
        
        Parameters:
        - layer: The layer index (integer) for which to perform the chi-square tests.
        - positions: Optional list of weight positions (tuples: (neuron_idx, from_weight_idx)) to test.
                     If None, the test will run for every weight position in the specified layer.
        - apply_correction: Boolean flag; if True, applies multiple comparisons correction to the p-values.
        - correction_method: Method for correction, e.g. 'bonferroni', 'holm', or 'fdr_bh'.
        
        Returns:
        - results: Dictionary mapping each weight position to a dictionary containing:
            {
                "chi2_stat": <chi-square statistic>,
                "p_value": <raw p-value>,
                "residuals": <standardized residuals array for each bin>,
                "expected": <expected counts array>,
                "observed_ref": <observed counts from reference>,
                "observed_phen": <observed counts from phenotype>
            }
        """
        results = {}
        p_values = []
        positions_list = []

        # Get the distributions for the specified layer from both objects.
        ref_dist_layer = self.layer_weight_distributions_ref[layer]
        phen_dist_layer = self.layer_weight_distributions_phen[layer]
        
        num_neurons, num_from_weights, num_bins = ref_dist_layer.shape
        
        # Determine which positions to process.
        if positions is None:
            # Process all weight positions in this layer.
            positions = [(i, j) for i in range(num_neurons) for j in range(num_from_weights)]
        
        # Loop through specified weight positions
        for (i, j) in positions:
            # Extract 1D histograms for the weight position (length should be num_bins)
            counts_ref = ref_dist_layer[i, j, :]
            counts_phen = phen_dist_layer[i, j, :]
            
            # Skip if either distribution contains only None values (all-zero weights)
            if self.reference_WB_object.is_all_zero_weight(counts_ref) or self.phenotype_WB_object.is_all_zero_weight(counts_phen):
                continue
            
            # Run the chi-square test
            chi2_result = self._chi_square_test_for_position(counts_ref, counts_phen)
            
            # Skip if the test couldn't be performed
            if chi2_result[0] is None:
                continue
                
            chi2_stat, p_value, residuals, expected, merge_info = chi2_result
            
            results[(i, j)] = {
                "chi2_stat": chi2_stat,
                "p_value": p_value,
                "residuals": residuals,
                "expected": expected,
                "observed_ref": counts_ref,
                "observed_phen": counts_phen,
                "merge_info": merge_info
            }
            p_values.append(p_value)
            positions_list.append((i, j))
        
        # Optionally apply multiple comparisons correction
        if apply_correction and p_values:
            reject, pvals_corrected, _, _ = multipletests(p_values, alpha=self.chi2_alpha, method=correction_method)
            # Update results with corrected p-values and rejection decisions.
            for idx, pos in enumerate(positions_list):
                results[pos]["p_value_corrected"] = pvals_corrected[idx]
                results[pos]["reject_null"] = bool(reject[idx])
        
        return results
        
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
            
            # Skip if reference model is None (all-zero weights)
            if ref_model is None:
                continue
                
            for j, phen_cluster_id in enumerate(phen_clusters):
                phen_pos = phen_unique_clusters[phen_cluster_id]
                phen_model = phen_models[phen_pos[0], phen_pos[1]]
                
                # Skip if phenotype model is None (all-zero weights)
                if phen_model is None:
                    continue
                
                # Compute symmetrized KL divergence
                kl_forward = self.reference_WB_object.compute_kl_divergence(
                    ref_model, phen_model, multi_peak=True)
                kl_backward = self.reference_WB_object.compute_kl_divergence(
                    phen_model, ref_model, multi_peak=True)
                kl_divergence = (kl_forward + kl_backward) / 2
                
                # Store KL divergence in the comparison matrix
                comparison_matrix[i, j] = kl_divergence
                
                # Check if the distributions match based on KL threshold
                if kl_divergence < self.kl_match_threshold:
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
                # Skip if this is an all-zero weight in either reference or phenotype
                ref_distribution = self.layer_weight_distributions_ref[ref_cluster_indices.shape[0]][neuron_idx, from_weight_idx]
                phen_distribution = self.layer_weight_distributions_phen[phen_cluster_indices.shape[0]][neuron_idx, from_weight_idx]
                
                if (self.reference_WB_object.is_all_zero_weight(ref_distribution) or 
                    self.phenotype_WB_object.is_all_zero_weight(phen_distribution)):
                    continue
                
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
                is_match = kl_value < self.kl_match_threshold
                
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
        
        # Extract unique clusters from both models, filtering out all-zero weights
        ref_unique_clusters = self._extract_unique_clusters(ref_cluster_indices, layer, is_reference=True)
        phen_unique_clusters = self._extract_unique_clusters(phen_cluster_indices, layer, is_reference=False)
        
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
                   Additionally, mark those positions as "matched" if KL divergence < self.kl_match_threshold.
        """
        # Verify that both objects have cluster indices for the specified layer
        assert hasattr(self.reference_WB_object, 'cluster_indices') and hasattr(self.phenotype_WB_object, 'cluster_indices'), \
            "Cluster indices not available. Run cluster_distributions() first."
        
        # Get the GMM models for the specified layer from both objects
        ref_models = self.reference_WB_object.gmm_models[layer]
        phen_models = self.phenotype_WB_object.gmm_models[layer]
        
        # Get the distributions for checking all-zero weights
        ref_distributions = self.layer_weight_distributions_ref[layer]
        phen_distributions = self.layer_weight_distributions_phen[layer]
        
        # Get the shape of the layer (number of neurons and weights)
        num_neurons, num_from_weights = ref_models.shape
        
        # Initialize results dictionary
        results = {}
        
        # Compare distributions for each weight position
        for neuron_idx in range(num_neurons):
            for from_weight_idx in range(num_from_weights):
                # Skip if this is an all-zero weight in either reference or phenotype
                if (self.reference_WB_object.is_all_zero_weight(ref_distributions[neuron_idx, from_weight_idx]) or 
                    self.phenotype_WB_object.is_all_zero_weight(phen_distributions[neuron_idx, from_weight_idx])):
                    continue
                
                # Get the models at this position
                ref_model = ref_models[neuron_idx, from_weight_idx]
                phen_model = phen_models[neuron_idx, from_weight_idx]
                
                # Skip if either model is None
                if ref_model is None or phen_model is None:
                    continue
                
                # Compute KL divergence between the distributions
                kl_forward = self.reference_WB_object.compute_kl_divergence(
                    ref_model, phen_model, multi_peak=True)
                kl_backward = self.reference_WB_object.compute_kl_divergence(
                    phen_model, ref_model, multi_peak=True)
                
                # Use symmetrized KL divergence (average of both directions)
                kl_divergence = (kl_forward + kl_backward) / 2
                
                # Determine if the distributions match based on KL threshold
                is_match = kl_divergence < self.kl_match_threshold
                
                # Store results in the dictionary
                results[(neuron_idx, from_weight_idx)] = {
                    'kl_divergence': kl_divergence,
                    'is_match': is_match
                }
        
        return results
    
    
    def _extract_unique_clusters(self, cluster_indices, layer=0, is_reference=True):
        """
        Extract unique clusters from a cluster indices matrix, filtering out all-zero weights.
        
        Parameters:
        - cluster_indices: 2D array of cluster indices
        - layer: Layer index for checking all-zero weights
        - is_reference: Boolean indicating if this is the reference object (True) or phenotype (False)
        
        Returns:
        - unique_clusters: Dict mapping cluster_id -> (neuron_idx, from_weight_idx)
        """
        unique_clusters = {}
        distributions = self.layer_weight_distributions_ref[layer] if is_reference else self.layer_weight_distributions_phen[layer]
        
        for neuron_idx in range(cluster_indices.shape[0]):
            for from_weight_idx in range(cluster_indices.shape[1]):
                # Skip if this is an all-zero weight
                if (is_reference and self.reference_WB_object.is_all_zero_weight(distributions[neuron_idx, from_weight_idx])) or \
                   (not is_reference and self.phenotype_WB_object.is_all_zero_weight(distributions[neuron_idx, from_weight_idx])):
                    continue
                
                cluster_id = cluster_indices[neuron_idx, from_weight_idx]
                if cluster_id not in unique_clusters:
                    # Store the first position we find for this cluster
                    unique_clusters[cluster_id] = (neuron_idx, from_weight_idx)
        return unique_clusters

    def plot_chi_square_comparison(self, layer, position, chi2_results, figsize=(20, 12), alpha=0.5, ylim=None, 
                                  show_merged_bins=True, show_gmm=True, residual_cmap='coolwarm', bin_cmap='viridis'):
        """
        Plot a detailed comparison of reference and phenotype weight distributions with chi-square test results.
        
        Parameters:
        - layer: Layer index
        - position: Tuple (neuron_idx, from_weight_idx) position to plot
        - chi2_results: Results dictionary from binwise_chi_square_comparison for this position
        - figsize: Figure size
        - alpha: Transparency for histogram bars
        - ylim: Y-axis limits (optional)
        - show_merged_bins: Whether to highlight merged bins
        - show_gmm: Whether to overlay GMM fits
        - residual_cmap: Colormap for residual values
        - bin_cmap: Colormap for merged bins
        
        Returns:
        - fig: Matplotlib figure
        """
        
        # Extract data for the specified position
        result = chi2_results.get(position)
        if result is None:
            print(f"No chi-square results for position {position} in layer {layer}")
            return None
        
        # Extract key information
        counts_ref = result["observed_ref"]
        counts_phen = result["observed_phen"]
        residuals = result["residuals"]
        merge_info = result.get("merge_info", (0, 0))
        chi2_stat = result["chi2_stat"]
        p_value = result["p_value"]
        
        # Get bin edges for this layer
        bin_edges = self.reference_WB_object.layer_bin_ranges[layer]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        
        # Filter out None values for plotting
        valid_ref = np.array([c if c is not None else 0 for c in counts_ref])
        valid_phen = np.array([c if c is not None else 0 for c in counts_phen])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Determine distribution type for title and y-axis label
        dist_type = "Probability" if self.use_normalized else "Count"
        
        # Plot histograms side by side
        ax1.set_title(f"Reference Distribution", fontsize=16)
        ax2.set_title(f"Phenotype Distribution", fontsize=16)
        
        # Color bars based on residuals to highlight statistically significant differences
        # Higher absolute residual = more contribution to chi-square statistic
        residual_colors = []
        abs_residuals = np.abs(residuals[0]) if len(residuals) > 0 and len(residuals[0]) > 0 else []
        max_residual = np.max(abs_residuals) if len(abs_residuals) > 0 else 1
        
        color_norm = mcolors.Normalize(vmin=-max_residual, vmax=max_residual)
        cmap = plt.cm.get_cmap(residual_cmap)
        
        # Original indices before merging
        k, m = merge_info
        orig_indices = list(range(len(counts_ref)))
        merged_indices = []
        
        if k > 1:
            merged_indices.extend(range(k))  # Lower tail merged bins
        if m > 1:
            merged_indices.extend(range(len(counts_ref) - m, len(counts_ref)))  # Upper tail merged bins
        
        # Calculate number of bins before and after merging
        original_bin_count = len(counts_ref)
        post_merge_bin_count = original_bin_count - (k if k > 1 else 0) - (m if m > 1 else 0) + (1 if k > 1 else 0) + (1 if m > 1 else 0)
        
        # Plot reference histogram
        bars1 = ax1.bar(bin_centers, valid_ref, width=bin_widths*0.9, alpha=alpha, edgecolor='black')
        
        # Plot phenotype histogram
        bars2 = ax2.bar(bin_centers, valid_phen, width=bin_widths*0.9, alpha=alpha, edgecolor='black')
        
        # Color bars based on residuals and add residual values above bars
        if len(residuals) > 0:
            # Determine maximum height for positioning residual text
            max_ref_height = np.max(valid_ref) if len(valid_ref) > 0 else 0
            max_phen_height = np.max(valid_phen) if len(valid_phen) > 0 else 0
            
            # Calculate text vertical position (slightly above each bar)
            text_offset_ref = max_ref_height * 0.03
            text_offset_phen = max_phen_height * 0.03
            
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                if i < len(residuals[0]):  # Check if index is valid for residuals
                    # Reference plots with phenotype residuals
                    phenotype_residual = residuals[1][i] if i < len(residuals[1]) else 0
                    residual_color = cmap(color_norm(phenotype_residual))
                    bar1.set_facecolor(residual_color)
                    bar1.set_alpha(0.7)
                    
                    # Add residual value text above the bar
                    if not np.isnan(phenotype_residual):
                        ax1.text(bar1.get_x() + bar1.get_width()/2, 
                                bar1.get_height() + text_offset_ref,
                                f"{phenotype_residual:.2f}",
                                ha='center', va='bottom', fontsize=10, 
                                rotation=45, fontweight='bold')
                    
                    # Phenotype plots with reference residuals
                    reference_residual = residuals[0][i] if i < len(residuals[0]) else 0
                    residual_color = cmap(color_norm(reference_residual))
                    bar2.set_facecolor(residual_color)
                    bar2.set_alpha(0.7)
                    
                    # Add residual value text above the bar
                    if not np.isnan(reference_residual):
                        ax2.text(bar2.get_x() + bar2.get_width()/2, 
                                bar2.get_height() + text_offset_phen,
                                f"{reference_residual:.2f}", 
                                ha='center', va='bottom', fontsize=10,
                                rotation=45, fontweight='bold')
        
        # Highlight merged bins
        if show_merged_bins and merged_indices:
            merged_cmap = plt.cm.get_cmap(bin_cmap)
            for i in merged_indices:
                if i < len(bars1):
                    bars1[i].set_hatch('///')
                    bars1[i].set_edgecolor('black')
                    bars2[i].set_hatch('///')
                    bars2[i].set_edgecolor('black')
        
        # Plot GMM fits if requested
        if show_gmm:
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
            
            # Reference GMM
            ref_gmm = self.reference_WB_object.gmm_models[layer][position[0], position[1]]
            if ref_gmm is not None:
                ref_pdf = np.exp(ref_gmm.score_samples(x_fit.reshape(-1, 1)))
                # Scale to match histogram height
                max_count_ref = np.max(valid_ref)
                scaling_factor_ref = max_count_ref / np.max(ref_pdf) if np.max(ref_pdf) > 0 else 1
                ax1.plot(x_fit, ref_pdf * scaling_factor_ref, 'r-', lw=2, label='GMM Fit')
                
                # Plot individual components
                for i, (w, m, c) in enumerate(zip(ref_gmm.weights_, ref_gmm.means_, ref_gmm.covariances_)):
                    mu = m[0]
                    sigma = np.sqrt(c[0][0])
                    component_pdf = w * norm.pdf(x_fit, mu, sigma)
                    ax1.plot(x_fit, component_pdf * scaling_factor_ref, 'r--', alpha=0.5)
            
            # Phenotype GMM
            phen_gmm = self.phenotype_WB_object.gmm_models[layer][position[0], position[1]]
            if phen_gmm is not None:
                phen_pdf = np.exp(phen_gmm.score_samples(x_fit.reshape(-1, 1)))
                # Scale to match histogram height
                max_count_phen = np.max(valid_phen)
                scaling_factor_phen = max_count_phen / np.max(phen_pdf) if np.max(phen_pdf) > 0 else 1
                ax2.plot(x_fit, phen_pdf * scaling_factor_phen, 'b-', lw=2, label='GMM Fit')
                
                # Plot individual components
                for i, (w, m, c) in enumerate(zip(phen_gmm.weights_, phen_gmm.means_, phen_gmm.covariances_)):
                    mu = m[0]
                    sigma = np.sqrt(c[0][0])
                    component_pdf = w * norm.pdf(x_fit, mu, sigma)
                    ax2.plot(x_fit, component_pdf * scaling_factor_phen, 'b--', alpha=0.5)
        
        # Set axis limits
        if ylim:
            ax1.set_ylim(ylim)
            ax2.set_ylim(ylim)
        else:
            max_count = max(np.max(valid_ref), np.max(valid_phen))
            # Increase y-limit to make room for residual labels
            ax1.set_ylim(0, max_count * 1.25)
            ax2.set_ylim(0, max_count * 1.25)
        
        # Add legend and labels
        ax1.set_xlabel('Weight Value', fontsize=14)
        ax2.set_xlabel('Weight Value', fontsize=14)
        ax1.set_ylabel(dist_type, fontsize=14)
        
        # Add chi-square test results and KL divergence as text
        fig.suptitle(f"Chi-Square Comparison for Layer {layer}, Position {position} ({dist_type}s)", fontsize=18)
        text = f"Chi² = {chi2_stat:.2f}\np-value = {p_value:.4f}"
        if "p_value_corrected" in result:
            text += f"\nCorrected p-value = {result['p_value_corrected']:.4f}"
            text += f"\nReject null: {result['reject_null']}"
        
        # Add KL divergence if available
        try:
            ref_gmm = self.reference_WB_object.gmm_models[layer][position[0], position[1]]
            phen_gmm = self.phenotype_WB_object.gmm_models[layer][position[0], position[1]]
            if ref_gmm is not None and phen_gmm is not None:
                kl_forward = self.reference_WB_object.compute_kl_divergence(ref_gmm, phen_gmm, multi_peak=True)
                kl_backward = self.reference_WB_object.compute_kl_divergence(phen_gmm, ref_gmm, multi_peak=True)
                kl_divergence = (kl_forward + kl_backward) / 2
                text += f"\nKL divergence = {kl_divergence:.4f}"
        except:
            pass
        
        # Add a colorbar for residuals
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.1, shrink=0.5, label='Standardized Residuals')
        
        # Create a legend for merged bins with bin count information
        legend_elements = []
        if show_merged_bins and merged_indices:
            legend_elements.append(Patch(facecolor='white', edgecolor='black', hatch='///', label='Merged Bins'))
        
        # Add bin count information to legend
        legend_elements.append(Patch(facecolor='none', edgecolor='none', 
                                     label=f'Bins: {original_bin_count} → {post_merge_bin_count} (after merge)'))
        
        # Add if using normalized distributions to legend
        legend_elements.append(Patch(facecolor='none', edgecolor='none', 
                                    label=f'Using {"normalized" if self.use_normalized else "raw count"} distributions'))
        
        # Add the legend
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=12)
        
        # Adjust the subplot parameters to make room for the text and colorbar
        plt.subplots_adjust(top=0.85, bottom=0.25, left=0.1, right=0.9)
        
        # Add the text in a position that won't be cut off
        fig.text(0.5, 0.08, text, ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
        
        return fig

    def plot_significant_differences(self, layer, chi2_results, top_n=10, figsize=(12, 8), alpha=0.05):
        """
        Plot the most significant differences from a chi-square comparison.
        
        Parameters:
        - layer: Layer index
        - chi2_results: Results dictionary from binwise_chi_square_comparison
        - top_n: Number of most significant positions to plot
        - figsize: Figure size
        - alpha: Significance threshold
        
        Returns:
        - figs: List of matplotlib figures
        """
        
        # Filter positions by p-value
        if not chi2_results:
            print("No chi-square results provided.")
            return []
        
        # Sort positions by p-value (ascending)
        sorted_positions = sorted(
            chi2_results.items(), 
            key=lambda x: x[1].get("p_value_corrected", x[1]["p_value"])
        )
        
        # Take top_n most significant positions
        top_positions = sorted_positions[:top_n]
        
        figs = []
        for (pos, result) in top_positions:
            p_val = result.get("p_value_corrected", result["p_value"])
            if p_val < alpha:
                print(f"Plotting significant difference at position {pos}, p-value = {p_val:.4e}")
                fig = self.plot_chi_square_comparison(layer, pos, chi2_results)
                if fig:
                    figs.append(fig)
        
        if not figs:
            print(f"No positions with p-value < {alpha} found.")
        
        return figs

    def compare_and_plot(self, layer, positions=None, apply_correction=True, correction_method='fdr_bh', 
                        alpha=0.05, top_n=10, save_dir=None):
        """
        Perform chi-square comparison and plot the most significant differences.
        
        Parameters:
        - layer: Layer index to analyze
        - positions: List of positions to analyze (default: all)
        - apply_correction: Whether to apply multiple testing correction
        - correction_method: Method for multiple testing correction
        - alpha: Significance threshold
        - top_n: Number of most significant positions to plot
        - save_dir: Directory to save plots (default: None, don't save)
        
        Returns:
        - results: Chi-square comparison results
        - figs: List of matplotlib figures
        """
        
        # Run chi-square comparison
        results = self.binwise_chi_square_comparison(
            layer=layer,
            positions=positions,
            apply_correction=apply_correction,
            correction_method=correction_method
        )
        
        print(f"Performed chi-square comparison on {len(results)} weight positions")
        
        # Count significant results
        if apply_correction:
            significant = sum(1 for r in results.values() if r["p_value_corrected"] < alpha)
        else:
            significant = sum(1 for r in results.values() if r["p_value"] < alpha)
        
        print(f"Found {significant} positions with significant differences (p < {alpha})")
        
        # Plot significant differences
        figs = self.plot_significant_differences(
            layer=layer,
            chi2_results=results,
            top_n=top_n,
            alpha=alpha
        )
        
        # Save figures if requested
        if save_dir and figs:
            os.makedirs(save_dir, exist_ok=True)
            for i, fig in enumerate(figs):
                fig.savefig(os.path.join(save_dir, f"chi2_diff_{layer}_rank{i+1}.png"), dpi=300, bbox_inches='tight')
        
        return results, figs

    def set_distribution_type(self, use_normalized=None):
        """
        Change the distribution type used for comparisons.
        
        Parameters:
        - use_normalized: Boolean; if True, use normalized distributions, if False, use raw counts.
                          If None, toggle the current setting.
        
        Returns:
        - The current setting after the change.
        """
        if use_normalized is None:
            # Toggle current setting
            use_normalized = not self.use_normalized
            
        self.use_normalized = use_normalized
        
        # Update the distributions based on the new setting
        if self.use_normalized:
            # Ensure normalized distributions are available
            if not hasattr(self.reference_WB_object, 'normalized_distributions'):
                self.reference_WB_object.normalize_distributions()
            if not hasattr(self.phenotype_WB_object, 'normalized_distributions'):
                self.phenotype_WB_object.normalize_distributions()
                
            self.layer_weight_distributions_ref = self.reference_WB_object.normalized_distributions
            self.layer_weight_distributions_phen = self.phenotype_WB_object.normalized_distributions
            print("Switched to normalized distributions for comparison")
        else:
            # Use raw counts
            self.layer_weight_distributions_ref = self.reference_WB_object.layer_weight_distributions_counts
            self.layer_weight_distributions_phen = self.phenotype_WB_object.layer_weight_distributions_counts
            print("Switched to raw count distributions for comparison")
            
        return self.use_normalized

    def prepare_bin_data_for_bin(self, layer, weight_position, bin_index):
        """
        For a given layer and weight position (tuple: (neuron_idx, from_weight_idx)), 
        create a binary outcome vector for a single bin specified by bin_index.
        
        The outcome vector is generated separately for the reference and phenotype groups by 
        "expanding" the aggregated counts:
        - For example, if at the specified weight position the reference distribution shows
            that 20 out of 100 networks fell in bin_index, then we create a vector of length 100
            with 20 ones (indicating bin membership) and 80 zeros.
        
        A corresponding group-label vector is also created (0 for reference, 1 for phenotype).
        
        Returns:
            y: Combined binary outcome vector (1 = weight falls in the bin, 0 = otherwise)
            group_labels: Combined group labels vector (0 for reference, 1 for phenotype)
        
        """
        # Retrieve the aggregated count arrays for the given layer and weight position.
        # These arrays should be of shape (num_bins,).
        ref_counts = self.layer_weight_distributions_ref[layer][weight_position[0], weight_position[1], :]
        phen_counts = self.layer_weight_distributions_phen[layer][weight_position[0], weight_position[1], :]
        
        # Check if all values are None (indicating all-zero weights)
        if all(c is None for c in ref_counts) or all(c is None for c in phen_counts):
            # Return empty arrays to indicate this position should be skipped
            return np.array([]), np.array([])
        
        # Filter out None values for counts (replacing with zeros for calculations)
        ref_counts_clean = np.array([0 if c is None else c for c in ref_counts])
        phen_counts_clean = np.array([0 if c is None else c for c in phen_counts])
        
        # Determine the total number of networks in each group by summing the counts.
        total_ref = int(np.sum(ref_counts_clean))
        total_phen = int(np.sum(phen_counts_clean))
        
        # If either group has no data, return empty arrays
        if total_ref == 0 or total_phen == 0:
            return np.array([]), np.array([])
        
        # Get the count for the target bin. Check if it's None first to avoid errors
        if bin_index < len(ref_counts) and ref_counts[bin_index] is None:
            count_ref = 0
        else:
            count_ref = int(ref_counts_clean[bin_index])
        
        if bin_index < len(phen_counts) and phen_counts[bin_index] is None:
            count_phen = 0
        else:
            count_phen = int(phen_counts_clean[bin_index])
        
        # Create binary outcomes:
        # For the reference group: count_ref ones (bin membership) and the remaining zeros.
        outcome_ref = np.concatenate([np.ones(count_ref), np.zeros(total_ref - count_ref)])
        # For the phenotype group:
        outcome_phen = np.concatenate([np.ones(count_phen), np.zeros(total_phen - count_phen)])
        
        # Create corresponding group labels: 0 for reference, 1 for phenotype.
        labels_ref = np.zeros(total_ref)   # Reference group label 0.
        labels_phen = np.ones(total_phen)    # Phenotype group label 1.
        
        # Combine the outcomes and group labels from both groups.
        y = np.concatenate([outcome_ref, outcome_phen])
        group_labels = np.concatenate([labels_ref, labels_phen])
        
        return y, group_labels


    def _logistic_test_for_bin(self, y, group_labels):
        """
        Given the binary outcome vector (y) and the corresponding group labels for a single bin,
        fit a logistic regression model:
        
            logit(P(Bin Membership = 1)) = beta_0 + beta_1 * Group
        
        Extract the p-value for the group coefficient beta_1, as well as the odds ratio and 
        confidence interval.
        
        Returns:
            A dictionary with keys:
                'p_value': p-value for the group coefficient,
                'odds_ratio': exp(beta_1) indicating the effect size,
                'conf_int': Confidence interval for beta_1 (in exponentiated form, if desired),
                'model_summary': A summary of the logistic regression results.
        """
        
        # Add a constant term to the predictor for the intercept.
        X = sm.add_constant(group_labels)
        
        # Fit the logistic regression model.
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        
        # Extract the p-value for the group coefficient (the second coefficient, index 1).
        p_value = result.pvalues[1]
        # Calculate the odds ratio (exponentiating the coefficient).
        odds_ratio = np.exp(result.params[1])
        # Get the 95% confidence interval for beta_1 and exponentiate for an OR CI.
        conf_int = result.conf_int()
        
        # Check the type of conf_int and handle accordingly
        if hasattr(conf_int, 'iloc'):  # If it's a DataFrame
            conf_int = conf_int.iloc[1].tolist()
        else:  # If it's a NumPy array
            conf_int = conf_int[1].tolist()
        
        conf_int_exp = [np.exp(val) for val in conf_int]
        
        return {
            'p_value': p_value,
            'odds_ratio': odds_ratio,
            'conf_int': conf_int_exp,
            'model_summary': result.summary().as_text()
        }
    
    def binwise_logistic_regression_comparison(self, layer, positions=None, apply_correction=False, correction_method='fdr_bh'):
        """
        For a given layer (and optionally for specified weight positions), perform a logistic regression test
        for each bin at each weight position.
        
        For every weight position (or for the specified subset):
        - For each bin (from 0 to num_bins - 1):
            • Call prepare_bin_data_for_bin to obtain the binary outcome vector and group labels.
            • Call _logistic_test_for_bin to get the p-value, odds ratio, and other statistics.
        - Store these results in a dictionary keyed by weight position and bin index.
        - Optionally, collect and later apply multiple testing correction across all bins.
        
        Returns:
            results: A dictionary where each key is a weight position tuple (i, j) and the value is
                    another dictionary mapping bin index to the logistic regression results.
                    Example structure:
                    {
                        (i, j): {
                            0: {'p_value': ..., 'odds_ratio': ..., 'conf_int': ..., ...},
                            1: { ... },
                            ...
                        },
                        ...
                    }
            pval_list: A list of tuples ((i, j), bin_index, raw_p_value) for all tests performed.
        """
        results = {}
        pval_list = []
        
        # Get the shape of the layer from one of the distributions
        ref_layer = self.layer_weight_distributions_ref[layer]
        num_neurons, num_from_weights, num_bins = ref_layer.shape
        
        # Determine which positions to process
        if positions is None:
            positions = [(i, j) for i in range(num_neurons) for j in range(num_from_weights)]
        
        # Loop through specified weight positions
        for pos in positions:
            i, j = pos
            results[pos] = {}
            # Loop over each bin index
            for bin_idx in range(num_bins):
                # Prepare binary data for this bin at position (i,j)
                y, group_labels = self.prepare_bin_data_for_bin(layer, pos, bin_idx)
                # Skip if data is empty (e.g., no networks)
                if y.size == 0:
                    continue
                # Run logistic regression test for this bin
                test_result = self._logistic_test_for_bin(y, group_labels)
                results[pos][bin_idx] = test_result
                pval_list.append((pos, bin_idx, test_result['p_value']))
        
        # Optionally, if multiple testing correction is desired, call apply_logistic_correction.
        if apply_correction and pval_list:
            results = self.apply_logistic_correction(results, pval_list, correction_method=correction_method)
        
        return results, pval_list


    def apply_logistic_correction(self, results, pval_list, correction_method='fdr_bh'):
        """
        Given the list of p-values from all bins (or weight positions), apply multiple testing correction.
        
        Steps:
        - Extract all p-values from pval_list.
        - Use statsmodels.stats.multitest.multipletests to compute corrected p-values.
        - Update the results dictionary so that each test result includes:
                • 'p_value_corrected'
                • 'reject_null' flag (True if corrected p-value < self.chi2_alpha)
        
        Parameters:
            results: The nested dictionary of logistic regression results.
            pval_list: A list of tuples ((i, j), bin_index, raw_p_value).
            correction_method: Correction method to use (e.g., 'bonferroni', 'holm', 'fdr_bh').
        
        Returns:
            Updated results dictionary with corrected p-values.
        """

        # Extract the raw p-values in order
        raw_pvals = [item[2] for item in pval_list]
        
        # Apply multiple testing correction
        reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=self.chi2_alpha, method=correction_method)
        
        # Update the results dictionary with the corrected p-values and rejection flags
        for idx, (pos, bin_idx, raw_pval) in enumerate(pval_list):
            results[pos][bin_idx]['p_value_corrected'] = pvals_corrected[idx]
            results[pos][bin_idx]['reject_null'] = bool(reject[idx])
        
        return results


    def plot_logistic_regression_results(self, layer, weight_position, logistic_results, show_histogram=True):
        """
        Visualize the per-bin logistic regression results for a given weight position.
        
        For the specified weight position (tuple: (neuron_idx, from_weight_idx)) in the given layer,
        this function generates a Manhattan-style plot showing for each bin:
            - The -log10(p_value) (or -log10(corrected p_value) if available).
            - Optionally, the effect size (odds ratio) can be overlaid or plotted in a separate panel.
        If show_histogram is True, the original binned histogram for the reference and phenotype groups
        is overlaid to help visualize where the significant differences occur.
        
        Parameters:
            layer: The layer index.
            weight_position: Tuple (neuron_idx, from_weight_idx) position to plot.
            logistic_results: The results dictionary from binwise_logistic_regression_comparison for this position.
            show_histogram: Boolean flag to indicate whether to overlay the binned histogram.
        
        Returns:
            fig: A matplotlib figure containing the plot.
        """
        
        # Extract number of bins from one of the group's distributions
        bin_edges = self.reference_WB_object.layer_bin_ranges[layer]
        num_bins = len(bin_edges) - 1
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Prepare arrays to store -log10(p-values) and odds ratios for each bin.
        logp_values = np.empty(num_bins)
        odds_ratios = np.empty(num_bins)
        
        for bin_idx in range(num_bins):
            # Check if a test result exists for this bin; if not, set to NaN.
            if bin_idx in logistic_results:
                # Prefer corrected p-value if available
                p_val = logistic_results[bin_idx].get('p_value_corrected', logistic_results[bin_idx]['p_value'])
                logp_values[bin_idx] = -np.log10(p_val) if p_val > 0 else np.nan
                odds_ratios[bin_idx] = logistic_results[bin_idx]['odds_ratio']
            else:
                logp_values[bin_idx] = np.nan
                odds_ratios[bin_idx] = np.nan
        
        # Create the plot.
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot the -log10(p-value) per bin as bars.
        ax1.bar(bin_centers, logp_values, width=np.diff(bin_edges)*0.9, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Bin Range', fontsize=14)
        ax1.set_ylabel('-log10(p-value)', fontsize=14)
        ax1.set_title(f'Logistic Regression Association per Bin at Weight Position {weight_position} (Layer {layer})', fontsize=16)
        
        # If requested, overlay the binned histogram from the reference group.
        if show_histogram:
            # Retrieve the raw counts for the reference group at this weight position.
            ref_counts = self.reference_WB_object.layer_weight_distributions_counts[layer][weight_position[0], weight_position[1], :]
            # Overlay a line plot (scaled appropriately)
            ax2 = ax1.twinx()
            ax2.plot(bin_centers, ref_counts, color='red', marker='o', label='Reference Histogram')
            ax2.set_ylabel('Reference Count', fontsize=14, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
        
        # Add GWAS-style annotation: a horizontal line for a significance threshold if desired.
        # For example, if using a genome-wide threshold (this can be customized).
        significance_threshold = -np.log10(0.05)  # This is arbitrary; customize as needed.
        ax1.axhline(y=significance_threshold, color='gray', linestyle='--', label='Significance Threshold')
        ax1.legend(loc='upper left')
        
        plt.tight_layout()
        return fig

    def plot_distributions_log_reg_significance(self, layer, weight_position, logistic_results, figsize=(15, 7), alpha=0.8, 
                                              significance_threshold=0.05, cmap='Blues'):
        """
        Plot two histograms (reference and phenotype distributions) where the bins are colored 
        according to their statistical significance based on logistic regression results.
        
        Parameters:
            layer (int): The layer index to process.
            weight_position (tuple): Tuple (neuron_idx, from_weight_idx) position to plot.
            logistic_results (dict): Results dictionary from binwise_logistic_regression_comparison
                                    for the specified weight position.
            figsize (tuple): Figure size as (width, height).
            alpha (float): Opacity for histogram bars.
            significance_threshold (float): P-value threshold to consider a bin significant.
            cmap (str): Matplotlib colormap name to use for significant bins.
            
        Returns:
            fig: Matplotlib figure with the plot.
        """
        # Extract data for the specified position
        ref_counts = self.layer_weight_distributions_ref[layer][weight_position[0], weight_position[1], :]
        phen_counts = self.layer_weight_distributions_phen[layer][weight_position[0], weight_position[1], :]
        
        # Filter out None values
        ref_counts = np.array([0 if c is None else c for c in ref_counts])
        phen_counts = np.array([0 if c is None else c for c in phen_counts])
        
        # Get bin edges for this layer
        bin_edges = self.reference_WB_object.layer_bin_ranges[layer]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # Determine bin colors based on significance
        colormap = get_cmap(cmap)
        sig_color = colormap(0.8)  # Dark blue for significant bins
        
        # Normalize for non-significant bins (grayscale intensity)
        norm = mcolors.Normalize(vmin=0, vmax=-np.log10(significance_threshold))
        
        # Create bar plots with initial gray color
        bars1 = ax1.bar(bin_centers, ref_counts, width=bin_widths*0.9, 
                      color='gray', alpha=alpha, edgecolor='black')
        bars2 = ax2.bar(bin_centers, phen_counts, width=bin_widths*0.9, 
                      color='gray', alpha=alpha, edgecolor='black')
        
        # Set titles and labels
        ax1.set_title(f"Reference Distribution\nPosition {weight_position}", fontsize=14)
        ax2.set_title(f"Phenotype Distribution\nPosition {weight_position}", fontsize=14)
        ax1.set_xlabel("Weight Value", fontsize=12)
        ax2.set_xlabel("Weight Value", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        
        # Get the maximum height for text placement
        max_height = max(np.max(ref_counts), np.max(phen_counts))
        text_offset = max_height * 0.05
        
        # Find all significant bins first to calculate text positions
        significant_bin_indices = []
        for bin_idx in logistic_results:
            p_val = logistic_results[bin_idx].get('p_value_corrected', 
                                              logistic_results[bin_idx]['p_value'])
            if p_val < significance_threshold:
                significant_bin_indices.append(bin_idx)
        
        # Color bins based on significance and add p-values
        for bin_idx, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if bin_idx in logistic_results:
                # Get p-value (corrected if available)
                p_val = logistic_results[bin_idx].get('p_value_corrected', 
                                                      logistic_results[bin_idx]['p_value'])
                
                # Determine if significant
                is_significant = p_val < significance_threshold
                
                if is_significant:
                    # Color significant bins with the specified color for both distributions
                    bar1.set_facecolor(sig_color)
                    bar2.set_facecolor(sig_color)
                    
                    # Determine if there are adjacent significant bins to avoid text overlap
                    # Calculate vertical offset - alternate heights for adjacent significant bins
                    idx_in_sig_list = significant_bin_indices.index(bin_idx)
                    offset_multiplier = 1 + (idx_in_sig_list % 2) * 0.6  # Alternate between 1 and 1.6
                    
                    # Add p-value text above the bars
                    ax1.text(bar1.get_x() + bar1.get_width()/2, 
                           bar1.get_height() + text_offset * offset_multiplier,
                           f"p={p_val:.3e}", 
                           ha='center', va='bottom', fontsize=8, 
                           rotation=45, fontweight='bold')
                    
                    ax2.text(bar2.get_x() + bar2.get_width()/2, 
                           bar2.get_height() + text_offset * offset_multiplier,
                           f"p={p_val:.3e}", 
                           ha='center', va='bottom', fontsize=8, 
                           rotation=45, fontweight='bold')
                else:
                    # Use grayscale for non-significant bins, darker = closer to significance
                    if p_val > 0:  # Avoid log(0)
                        gray_intensity = norm(-np.log10(p_val))
                        gray_color = (gray_intensity, gray_intensity, gray_intensity)
                        bar1.set_facecolor(gray_color)
                        bar2.set_facecolor(gray_color)
    
        # Create legend for significance
        significant_patch = mpatches.Patch(color=sig_color, label=f'Significant (p < {significance_threshold})')
        nonsig_patch = mpatches.Patch(color='gray', label='Non-significant')
        
        # Add significance legend
        fig.legend(handles=[significant_patch, nonsig_patch], 
                 loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
        
        # Add a colorbar for grayscale p-values
        # Create a ScalarMappable for the grayscale norm
        gray_cmap = plt.cm.gray_r  # Reversed gray colormap (darker means more significant)
        sm = plt.cm.ScalarMappable(cmap=gray_cmap, norm=norm)
        sm.set_array([])  # Dummy array for the mappable
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=20)
        
        # Add tick at significance threshold
        cbar.ax.axhline(y=-np.log10(significance_threshold), color='red', linestyle='--', linewidth=1)
        cbar.ax.text(0.5, -np.log10(significance_threshold) + 0.1, 'threshold', 
                   color='red', ha='center', va='bottom', transform=cbar.ax.get_yaxis_transform())
        
        # Add KL divergence info if available
        try:
            ref_gmm = self.reference_WB_object.gmm_models[layer][weight_position[0], weight_position[1]]
            phen_gmm = self.phenotype_WB_object.gmm_models[layer][weight_position[0], weight_position[1]]
            if ref_gmm is not None and phen_gmm is not None:
                kl_forward = self.reference_WB_object.compute_kl_divergence(ref_gmm, phen_gmm, multi_peak=True)
                kl_backward = self.reference_WB_object.compute_kl_divergence(phen_gmm, ref_gmm, multi_peak=True)
                kl_divergence = (kl_forward + kl_backward) / 2
                
                # Add KL divergence as text on plot
                fig.text(0.5, 0.01, f"KL Divergence = {kl_divergence:.4f}", 
                        ha='center', fontsize=12)
        except:
            pass
        
        # Add a note about odds ratios
        significant_bins = [bin_idx for bin_idx in logistic_results 
                            if logistic_results[bin_idx].get('p_value_corrected', 
                                                            logistic_results[bin_idx]['p_value']) < significance_threshold]
        
        if significant_bins:
            # Find the bin with the smallest p-value
            min_pval_bin = min(significant_bins, 
                              key=lambda bin_idx: logistic_results[bin_idx].get('p_value_corrected', 
                                                                               logistic_results[bin_idx]['p_value']))
            odds_ratio = logistic_results[min_pval_bin]['odds_ratio']
            conf_int = logistic_results[min_pval_bin]['conf_int']
            
            fig.text(0.5, 0.03, 
                    f"Most significant bin: {min_pval_bin}, Odds Ratio = {odds_ratio:.2f} (95% CI: {conf_int[0]:.2f}-{conf_int[1]:.2f})",
                    ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15, right=0.9)  # Make room for the legend and colorbar
        
        return fig

    def plot_logistic_regression_curve(self, layer, weight_position, bin_index, logistic_result, figsize=(10, 6), use_log_scale=False):
        """
        Plot the logistic regression curve for a specific bin, showing the probability of a weight 
        falling into this bin as a function of group (reference vs phenotype).
        
        Parameters:
            layer (int): The layer index.
            weight_position (tuple): Tuple (neuron_idx, from_weight_idx) position.
            bin_index (int): Index of the bin to plot.
            logistic_result (dict): Dictionary containing the logistic regression results for this bin.
            figsize (tuple): Figure size as (width, height).
            
        Returns:
            fig: Matplotlib figure with the logistic regression curve.
        """
        
        # Get bin edges for this layer
        bin_edges = self.reference_WB_object.layer_bin_ranges[layer]
        bin_range = (bin_edges[bin_index], bin_edges[bin_index+1])
        
        # Prepare data for visualization
        # Get original binary data for this bin
        y, group_labels = self.prepare_bin_data_for_bin(layer, weight_position, bin_index)
        
        # Extract model parameters from logistic result
        X = sm.add_constant(group_labels)
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        
        # Get model parameters
        intercept = result.params[0]
        slope = result.params[1]
        odds_ratio = np.exp(slope)
        ci_low, ci_high = logistic_result['conf_int']
        p_value = logistic_result.get('p_value_corrected', logistic_result['p_value'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the actual proportions
        # For reference group (0)
        ref_data = y[group_labels == 0]
        ref_prop = np.mean(ref_data)
        ref_count = len(ref_data)
        
        # For phenotype group (1)
        phen_data = y[group_labels == 1]
        phen_prop = np.mean(phen_data)
        phen_count = len(phen_data)
        
        # Calculate standard errors for proportions
        ref_se = np.sqrt(ref_prop * (1 - ref_prop) / ref_count)
        phen_se = np.sqrt(phen_prop * (1 - phen_prop) / phen_count)
        
        # Determine if we should use a log scale
        max_prop = max(ref_prop, phen_prop)
        min_prop = min(max(0.001, ref_prop), max(0.001, phen_prop))  # Avoid 0 values for log scale
        prop_ratio = max_prop / min_prop if min_prop > 0 else 1
        
        if use_log_scale:
            # For log scale we need to ensure no zero values
            ax.set_yscale('log')
            # Ensure positive lower limit for log scale
            ax.set_ylim(bottom=max(0.0005, min_prop/2), top=0.3)
            
            # Add grid with appropriate y-axis minor ticks
            ax.grid(True, which='major', linestyle='--', alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            
            # Set custom formatter to show probabilities nicely
            from matplotlib.ticker import FuncFormatter
            def probability_formatter(x, pos):
                # Format as percentage for small numbers
                if x < 0.01:
                    return f"{x:.1e}"
                else:
                    return f"{x:.2f}"
            
            ax.yaxis.set_major_formatter(FuncFormatter(probability_formatter))
        
        # Plot observed proportions with error bars (after setting scale)
        ax.errorbar([0, 1], [ref_prop, phen_prop], yerr=[ref_se, phen_se], 
                  fmt='o', capsize=5, label='Observed proportion', markersize=8)
        
        # Plot the logistic regression curve
        x_curve = np.linspace(-0.2, 1.2, 100)
        X_curve = sm.add_constant(x_curve)
        y_curve = expit(X_curve @ np.array([intercept, slope]))
        
        ax.plot(x_curve, y_curve, 'r-', label='Logistic regression curve')
        
        # Calculate confidence bands for the curve
        # Get covariance matrix
        cov_matrix = result.cov_params()
        
        # Standard error for the linear predictor
        se_band = np.zeros_like(x_curve)
        for i, x_val in enumerate(x_curve):
            x_with_const = np.array([1, x_val])
            se_band[i] = np.sqrt(x_with_const.T @ cov_matrix @ x_with_const)
        
        # 95% confidence intervals for the linear predictor
        lower_bound = X_curve @ np.array([intercept, slope]) - 1.96 * se_band
        upper_bound = X_curve @ np.array([intercept, slope]) + 1.96 * se_band
        
        # Transform to probability scale
        lower_curve = expit(lower_bound)
        upper_curve = expit(upper_bound)
        
        # Plot confidence bands
        ax.fill_between(x_curve, lower_curve, upper_curve, color='red', alpha=0.2, label='95% CI')
        
        # Set axis labels and title
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Reference', 'Phenotype'], fontsize=12)
        ax.set_ylabel('Probability of weight in bin', fontsize=12)
        
        # Set y-axis limits if not using log scale
        if not use_log_scale:
            # Better Y-axis scaling for visual appeal
            # Calculate maximum y value including confidence intervals
            max_y_data = max(np.max(upper_curve), ref_prop + ref_se, phen_prop + phen_se)
            
            # If the max value is small, use a more focused range rather than going to 1.0
            if max_y_data < 0.5:
                # Add padding and round to nice number 
                y_max = min(1.0, np.ceil(max_y_data * 6) / 10)  # Round to nearest 0.1 above the data and add 20% margin
                ax.set_ylim(0, y_max)
                
                # Add more y-axis ticks for better visual reference when range is small
                ax.set_yticks(np.linspace(0, y_max, min(6, int(y_max * 10) + 1)))
            else:
                # For larger proportions, use the original approach
                ax.set_ylim(0, max(1.0, max_y_data * 1.2))
        
        # Add title
        title = (f"Bin {bin_index} ({bin_range[0]:.4f} to {bin_range[1]:.4f})\n"
                f"Logistic Regression: p={p_value:.6f}, OR={odds_ratio:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f})")
        
        if use_log_scale:
            title += "\n(Log scale)"
            
        ax.set_title(title, fontsize=14)
        
        # Add counts and proportions as text
        y_offset = 0.05 * ax.get_ylim()[1] if not use_log_scale else ref_prop * 0.1
        ax.text(0, ref_prop + y_offset, f"n={ref_count}\n{ref_prop:.3f}", ha='center', va='bottom', fontsize=10)
        ax.text(1, phen_prop + y_offset, f"n={phen_count}\n{phen_prop:.3f}", ha='center', va='bottom', fontsize=10)
        
        # Add interpretation of odds ratio
        if odds_ratio > 1:
            interp_text = f"{odds_ratio:.2f}× higher odds in phenotype"
        else:
            interp_text = f"{1/odds_ratio:.2f}× lower odds in phenotype"
        
        ax.text(0.5, 0.05, interp_text, ha='center', transform=ax.transAxes, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Add legend
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig

    def plot_all_matching_distributions_with_significance(self, layer, save_dir=None, apply_correction=True, correction_method='fdr_bh'):
        """
        Driver function to create significance-colored histogram plots for all matching weight distributions.
        
        Parameters:
            layer (int): The layer index to process.
            save_dir (str, optional): If provided, save each plot to this directory.
            apply_correction (bool): Whether to apply multiple testing correction to p-values.
            correction_method (str): Method for multiple testing correction (e.g., 'fdr_bh', 'bonferroni').
            
        Returns:
            figs (list): List of matplotlib figure objects (one per matching weight position).
        """
        # 1. Identify matching weight positions
        matching_info = self.find_matching_unique_distributions(layer)
        # Filter out only the positions where the distributions match (i.e. is_match == True)
        matching_positions = [pos for pos, info in matching_info.items() if info['is_match']]
        print(f"Found {len(matching_positions)} matching weight positions in layer {layer}.")
        
        # 2. Run binwise logistic regression for only these matching weight positions
        logistic_results, pval_list = self.binwise_logistic_regression_comparison(
            layer=layer,
            positions=matching_positions,
            apply_correction=apply_correction,
            correction_method=correction_method
        )
        
        # 3. Plot the distributions with significance coloring
        figs = []
        for pos in matching_positions:
            # Retrieve the logistic regression results for this weight position
            pos_results = logistic_results.get(pos, {})
            if not pos_results:
                continue  # Skip if no results are available
            
            fig = self.plot_distributions_log_reg_significance(
                layer=layer,
                weight_position=pos,
                logistic_results=pos_results
            )
            figs.append(fig)
            
            # Optionally, save the figure if a save directory is provided
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                # Create a filename using the layer index and weight position
                filename = os.path.join(save_dir, f"significance_hist_layer{layer}_pos{pos[0]}_{pos[1]}.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved plot for weight position {pos} to {filename}")
        
        return figs
        
    def driver_logistic_regression_for_matching(self, layer, save_dir=None, include_original_plots=True, 
                                               apply_correction=True, correction_method='fdr_bh', 
                                               plot_logistic_curves=True):
        """
        Driver function to test each bin for the 'matching' weight positions in a given layer using
        a GWAS-inspired logistic regression approach. For each matching weight position (as determined by
        the KL divergence matching), it runs the binwise logistic regression tests and then plots both:
        1. The Manhattan-style -log10(p-value) plot 
        2. The significance-colored distribution histograms
        3. Logistic regression curves for significant bins

        Parameters:
            layer (int): The layer index to process.
            save_dir (str, optional): If provided, save each plot to this directory.
            include_original_plots (bool): Whether to include the original Manhattan-style plots.
            apply_correction (bool): Whether to apply multiple testing correction to p-values.
            correction_method (str): Method for multiple testing correction (e.g., 'fdr_bh', 'bonferroni').
            plot_logistic_curves (bool): Whether to generate logistic regression curves for significant bins.

        Returns:
            logistic_results (dict): Nested dictionary of logistic regression results keyed by weight position and bin.
            figs (list): List of matplotlib figure objects with both plot types.
            summary (dict): Summary of significant bins and their statistics.
        """
        # 1. Identify matching weight positions using your existing KL-based matching.
        matching_info = self.find_matching_unique_distributions(layer)
        # Filter out only the positions where the distributions match (i.e. is_match == True)
        matching_positions = [pos for pos, info in matching_info.items() if info['is_match']]
        print(f"Found {len(matching_positions)} matching weight positions in layer {layer}.")
        
        # 2. Run binwise logistic regression for only these matching weight positions.
        logistic_results, pval_list = self.binwise_logistic_regression_comparison(
            layer=layer,
            positions=matching_positions,
            apply_correction=apply_correction,
            correction_method=correction_method
        )
        
        # Initialize summary dictionary
        summary = {
            'positions_analyzed': len(matching_positions),
            'positions_with_significant_bins': 0,
            'total_significant_bins': 0,
            'positions_summary': {},
            'significance_threshold': self.chi2_alpha
        }
        
        # Get bin edges for readable weight values
        bin_edges = self.reference_WB_object.layer_bin_ranges[layer]
        
        # 3. Plot the results for each matching weight position using both visualization methods.
        figs = []
        for pos in matching_positions:
            # Retrieve the logistic regression results for this weight position.
            pos_results = logistic_results.get(pos, {})
            if not pos_results:
                continue  # Skip if no results are available
            
            # Find significant bins for this position
            significant_bins = []
            for bin_idx, result in pos_results.items():
                p_val = result.get('p_value_corrected', result['p_value'])
                if p_val < self.chi2_alpha:
                    significant_bins.append({
                        'bin_index': bin_idx,
                        'p_value': p_val,
                        'odds_ratio': result['odds_ratio'],
                        'confidence_interval': result['conf_int'],
                        'weight_range': (bin_edges[bin_idx], bin_edges[bin_idx+1])
                    })
            
            # Update summary
            if significant_bins:
                summary['positions_with_significant_bins'] += 1
                summary['total_significant_bins'] += len(significant_bins)
                summary['positions_summary'][str(pos)] = {
                    'position': pos,
                    'significant_bins': significant_bins,
                    'total_significant_bins': len(significant_bins)
                }
            
            # Create the new significance-colored distribution histograms
            sig_fig = self.plot_distributions_log_reg_significance(
                layer=layer,
                weight_position=pos,
                logistic_results=pos_results
            )
            figs.append(sig_fig)
            
            # Optionally save the significance plot
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                sig_filename = os.path.join(save_dir, f"significance_hist_layer{layer}_pos{pos[0]}_{pos[1]}.png")
                sig_fig.savefig(sig_filename, dpi=300, bbox_inches='tight')
                print(f"Saved significance plot for weight position {pos} to {sig_filename}")
            
            # Optionally create the original Manhattan-style plots
            if include_original_plots:
                manhattan_fig = self.plot_logistic_regression_results(
                    layer=layer,
                    weight_position=pos,
                    logistic_results=pos_results,
                    show_histogram=True
                )
                figs.append(manhattan_fig)
                
                # Save the Manhattan plot if requested
                if save_dir is not None:
                    manhattan_filename = os.path.join(save_dir, f"logistic_regression_layer{layer}_pos{pos[0]}_{pos[1]}.png")
                    manhattan_fig.savefig(manhattan_filename, dpi=300, bbox_inches='tight')
                    print(f"Saved Manhattan plot for weight position {pos} to {manhattan_filename}")
            
            # Plot logistic regression curves for significant bins
            if plot_logistic_curves and significant_bins:
                # Sort bins by p-value (ascending)
                sorted_bins = sorted(significant_bins, key=lambda x: x['p_value'])
                
                for bin_info in sorted_bins:
                    bin_idx = bin_info['bin_index']
                    bin_result = pos_results[bin_idx]
                    
                    # Create logistic regression curve plot
                    logistic_fig = self.plot_logistic_regression_curve(
                        layer=layer,
                        weight_position=pos,
                        bin_index=bin_idx,
                        logistic_result=bin_result,
                        use_log_scale=False
                    )
                    figs.append(logistic_fig)
                    
                    # Save the logistic regression curve plot if requested
                    if save_dir is not None:
                        logistic_filename = os.path.join(
                            save_dir, 
                            f"logistic_curve_layer{layer}_pos{pos[0]}_{pos[1]}_bin{bin_idx}.png"
                        )
                        logistic_fig.savefig(logistic_filename, dpi=300, bbox_inches='tight')
                        print(f"Saved logistic regression curve for position {pos}, bin {bin_idx} to {logistic_filename}")
        
        # Generate a text summary report
        summary_text = self._generate_significance_summary(summary, layer)
        print(summary_text)
        
        # Save the summary to a text file if save_dir is provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            summary_filename = os.path.join(save_dir, f"significance_summary_layer{layer}.txt")
            with open(summary_filename, 'w') as f:
                f.write(summary_text)
            print(f"Saved significance summary to {summary_filename}")
        
        return logistic_results, figs, summary

    def _generate_significance_summary(self, summary, layer):
        """
        Generate a formatted text summary of significant bins.
        
        Parameters:
            summary (dict): Dictionary containing summary statistics
            layer (int): Layer index
            
        Returns:
            str: Formatted text summary
        """
        summary_text = f"===== LOGISTIC REGRESSION SIGNIFICANCE SUMMARY FOR LAYER {layer} =====\n\n"
        summary_text += f"Total positions analyzed: {summary['positions_analyzed']}\n"
        summary_text += f"Positions with significant bins: {summary['positions_with_significant_bins']}\n"
        summary_text += f"Total significant bins across all positions: {summary['total_significant_bins']}\n"
        summary_text += f"Significance threshold: {summary['significance_threshold']}\n\n"
        
        if summary['positions_with_significant_bins'] > 0:
            summary_text += "--- POSITIONS WITH SIGNIFICANT BINS ---\n\n"
            
            # Sort positions by number of significant bins (descending)
            sorted_positions = sorted(
                summary['positions_summary'].items(),
                key=lambda x: x[1]['total_significant_bins'],
                reverse=True
            )
            
            for pos_key, pos_info in sorted_positions:
                pos = pos_info['position']
                summary_text += f"Position {pos} - {pos_info['total_significant_bins']} significant bins:\n"
                
                # Sort bins by p-value (ascending)
                sorted_bins = sorted(pos_info['significant_bins'], key=lambda x: x['p_value'])
                
                for bin_info in sorted_bins:
                    bin_idx = bin_info['bin_index']
                    p_val = bin_info['p_value']
                    odds_ratio = bin_info['odds_ratio']
                    ci_low, ci_high = bin_info['confidence_interval']
                    weight_range = bin_info['weight_range']
                    
                    summary_text += f"  * Bin {bin_idx} (weight range: {weight_range[0]:.4f} to {weight_range[1]:.4f}):\n"
                    summary_text += f"    - p-value: {p_val:.8f}\n"
                    summary_text += f"    - Odds ratio: {odds_ratio:.4f} (95% CI: {ci_low:.4f}-{ci_high:.4f})\n"
                    
                    # Interpret the odds ratio
                    if odds_ratio > 1:
                        summary_text += f"    - Interpretation: {odds_ratio:.2f}x higher odds of weights falling in this bin in phenotype vs. reference\n"
                    else:
                        summary_text += f"    - Interpretation: {1/odds_ratio:.2f}x lower odds of weights falling in this bin in phenotype vs. reference\n"
                    
                summary_text += "\n"
        else:
            summary_text += "No significant bins found in any position.\n"
            
        return summary_text