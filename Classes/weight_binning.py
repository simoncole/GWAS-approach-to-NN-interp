import os 
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from scipy.stats import norm



class WeightBinning():

    def __init__(self, architecture, save_dir, load_path, num_bins=30):

        # Assert that num_bins is positive
        assert num_bins > 0, "Number of bins must be positive"
        # Assert that load_path exists
        assert os.path.exists(load_path), f"Load path '{load_path}' does not exist"

        os.makedirs(save_dir, exist_ok=True) 
        self.num_bins = num_bins
        self.save_dir = save_dir
        self.load_path = load_path
        self.architecture = architecture
        self.load_models()
        print("Loaded models")
        self.store_weights()
        print("Stored weights")
        


    def __getMinOrMax__(self, networks, layerNum, getMin):
        """
        Parameters:
        - networks: an array of trained pytorch networks (works with both regular and permutation-free)
        - layerNum: an int representing the fully connected layer to access
        - getMin: a boolean indicating whether to get the min or max
        Outputs: the minimum or maximum weight in some array of neural networks
        """
        weights = []
        for network in networks:
            # Check if network has a nested 'network' attribute (like PermutationFreeNet)
            if hasattr(network, "network"):
                net_layers = list(network.network.children())
            else:
                net_layers = list(network.children())
            
            fc_layer_idx = 0  # Counter for Linear layers
            for layer in net_layers:
                if isinstance(layer, nn.Linear):
                    if fc_layer_idx == layerNum:
                        weights.extend(layer.weight.data.cpu().numpy().flatten())
                    fc_layer_idx += 1
        
        if getMin:
            return min(weights) - 1e-6 #small adjustment to account for floating point error
        else:
            return max(weights) + 1e-6
        
    def get_layer_weight_distributions(self):
        """
        Get the layer weight distributions data structure.
        
        Returns:
        - layer_weight_distributions_counts: A list of 3D numpy arrays, one per layer, with shape 
        (num neurons, num neurons in previous layer, num bins), where each bin contains the count of networks 
        whose weight at that position falls within the bin's range.
        
        Raises:
        - AttributeError: If store_weights() hasn't been called yet
        """
        if not hasattr(self, 'layer_weight_distributions_counts'):
            raise AttributeError("layer_weight_distributions_counts not available. Call store_weights() first.")
    
        return self.layer_weight_distributions_counts
        
    def load_models(self):
        #Load in the networks
        loaded_state_dict = torch.load(self.load_path, weights_only=True, map_location=torch.device('cpu'))
        
        # Assert that state_dict contains expected networks
        assert len(loaded_state_dict) > 0, "State dict is empty - no networks found"
        
        #create new instantiations to contain the saved networks
        num_networks = len(loaded_state_dict)
        self.networks = []
        for n in range(num_networks):
            self.networks.append(self.architecture())

        for i, network in enumerate(self.networks):
            state_dict_key = f'network_{i+1}'
            # Assert that the expected key exists in the state dict
            assert state_dict_key in loaded_state_dict, f"Expected key '{state_dict_key}' not found in state dict"
            network.load_state_dict(loaded_state_dict[state_dict_key])
        
        # Assert we have at least one network loaded
        assert len(self.networks) > 0, "No networks were loaded"
        
        print("Number of networks loaded:")
        print(len(self.networks))
        return self.networks

    def store_network_weights(self, network_fc_indices):
        """
        Extract weight data from all networks for the specified fully connected layers.
        
        Parameters:
        - network_fc_indices: List of indices corresponding to fully connected layers
        
        Returns:
        - network_weights: A list of lists, where each inner list contains the weight matrices
          for a specific layer across all networks
        """
        # Assert network_fc_indices is not empty
        assert len(network_fc_indices) > 0, "No fully connected layer indices provided"
        # Assert networks are loaded
        assert hasattr(self, 'networks') and len(self.networks) > 0, "Networks not loaded"
        
        network_weights = []
        for index, layer_index in enumerate(network_fc_indices):
            network_weights_per_layer = []
            for network in self.networks:
                if hasattr(network, "network"):
                    net_layers = list(network.network.children())
                else:
                    net_layers = list(network.children())
                
                network_weights_per_layer.append(net_layers[layer_index].weight.data.numpy())
            network_weights.append(network_weights_per_layer)
        
        return network_weights
    
    def bin_single_weight(self, weight_position, network_weights_layer, bin_ranges):
        """
        Bin a single weight position across a set of networks.
        
        Parameters:
        - weight_position: Tuple (neuron_idx, from_weight_idx)
        - network_weights_layer: Weight matrices for this layer across all networks, shape [networks][neurons][weights]
        - bin_ranges: Bin edge values for this layer
        
        Returns:
        - binned_networks: A list of lists where each inner list contains networks falling into a specific bin
                          for this weight position. Shape is [num_bins][networks_per_bin].
        """
        # Initialize empty lists for each bin
        binned_networks = [[] for _ in range(self.num_bins)]
        
        for net_idx, network in enumerate(network_weights_layer):
            # Select network[neuron i, incoming weight j]
            weight = network[weight_position[0]][weight_position[1]]
            corresponding_bin = np.digitize(weight, bin_ranges, right=False) - 1
            
            # Check if weight is in bin range
            assert corresponding_bin >= 0 and corresponding_bin < self.num_bins, f"Weight {weight} out of bin range at neuron {weight_position[0]}, incoming weight {weight_position[1]}"
            
            # Add network to the appropriate bin
            binned_networks[corresponding_bin].append(network)
        
        return binned_networks
    def populate_bins_single_layer(self, layer_num, layer_networks_skeleton, network_weights_layer, bin_ranges):
        """
        Populate the weight distribution bins for a single layer with networks whose weights fall into each bin.
        
        Parameters:
        - layer_num: Index of the layer being processed
        - layer_networks_skeleton: Empty structure for the layer, shape [neurons][incoming_weights][num_bins]
        - network_weights_layer: Weight matrices for this layer across all networks, shape [networks][neurons][weights]
        - bin_ranges: Bin edge values for this layer
        
        Returns:
        - layer_networks: The filled network distribution structure for this layer
        
        Notes:
        - Each bin contains a list of networks whose weights fall within that bin's range.
        """
        layer_networks = layer_networks_skeleton.copy()
        
        # Iterate over neurons in layer
        for i in range(layer_networks.shape[0]):
            # Iterate over incoming weights to neuron
            for j in range(layer_networks.shape[1]):
                # Bin the single weight position
                binned_networks = self.bin_single_weight((i, j), network_weights_layer, bin_ranges)
                
                # Store the binned networks in the layer structure
                for bin_idx, networks in enumerate(binned_networks):
                    layer_networks[i][j][bin_idx] = networks
        
        return layer_networks
    
    def populate_all_bins(self, network_weights, layer_weight_distributions_networks_skeleton):
        """
        Populate the weight distribution bins for all layers with networks whose weights fall into each bin.
        
        This function takes the weight data from the set of networks and categorizes each weight
        into the appropriate bin based on its value, storing the network in the corresponding bin.
        
        Parameters:
        - network_weights: A list of lists, where each inner list contains weight matrices
          for a specific layer across all networks. Shape is [layers][networks][neurons][weights].
        - layer_weight_distributions_networks_skeleton: An initialized empty structure for storing
          networks, with shape [layers][neurons][incoming_weights][num_bins].
        
        Returns:
        - layer_weight_distributions_networks: The filled distribution structure with the networks
          whose weights fall into each bin for each position in each layer.
        
        Notes:
        - Requires self.layer_bin_ranges to be calculated beforehand.
        - Each bin contains a list of networks whose weights fall within that bin's range.
        """
        # layer bin ranges should exist and have the same number of layers as the skeleton
        assert len(self.layer_bin_ranges) == len(layer_weight_distributions_networks_skeleton)

        layer_weight_distributions_networks = []
        
        # Process each layer
        for layer_num, layer_skeleton in enumerate(layer_weight_distributions_networks_skeleton):
            layer_networks = self.populate_bins_single_layer(
                layer_num=layer_num,
                layer_networks_skeleton=layer_skeleton,
                network_weights_layer=network_weights[layer_num],
                bin_ranges=self.layer_bin_ranges[layer_num]
            )
            layer_weight_distributions_networks.append(layer_networks)
                        
        return layer_weight_distributions_networks
    
    def derive_counts_from_networks(self, layer_weight_distributions_networks):
        """
        Derive count distributions from network distributions by counting networks in each bin.
        
        Parameters:
        - layer_weight_distributions_networks: The distribution structure with networks
          stored in bins, with shape [layers][neurons][incoming_weights][num_bins].
        
        Returns:
        - layer_weight_distributions_counts: A distribution structure with counts of networks
          in each bin, with shape [layers][neurons][incoming_weights][num_bins].
        """
        layer_weight_distributions_counts = []
        
        for layer in layer_weight_distributions_networks:
            layer_shape = layer.shape[:-1]  # Remove the bin dimension
            layer_counts = np.zeros(layer_shape + (self.num_bins,), dtype=int)
            
            for i in range(layer_shape[0]):
                for j in range(layer_shape[1]):
                    for bin_idx in range(self.num_bins):
                        # Count networks in this bin
                        layer_counts[i, j, bin_idx] = len(layer[i, j, bin_idx])
            
            layer_weight_distributions_counts.append(layer_counts)
            
        return layer_weight_distributions_counts
    
    def create_layer_skeleton(self, layer_shape):
        """
        Create an empty skeleton structure for a single layer's weight distribution.
        
        Parameters:
        - layer_shape: Shape of the layer's weight matrix (neurons, incoming_weights)
        
        Returns:
        - layer_skeleton: An initialized empty structure for storing networks,
          with shape [neurons][incoming_weights][num_bins] where each element is an empty list
        """
        # Create empty array with objects
        layer_skeleton = np.empty(layer_shape + (self.num_bins,), dtype=object)
        
        # Initialize each element to be an empty list
        for index in np.ndindex(layer_skeleton.shape):
            layer_skeleton[index] = []
            
        return layer_skeleton
        
    def store_weights(self):
        """
        Stores and bins the weight distributions loaded networks for further analysis.
        This method processes all networks which were loaded during initialization and extracts weight 
        values from each linear layer.
        
        The method stores the following class attributes:
        - layer_weight_distributions_networks: A list of 4D numpy arrays, one per layer, with shape 
          (num neurons, num neurons in previous layer, num bins), where each bin contains a list of networks
          whose weight at that position falls within the bin's range
        - layer_weight_distributions_counts: A list of 4D numpy arrays derived from networks, with counts
          of networks in each bin
        - layer_bin_ranges: A list of numpy arrays containing the bin edges for each layer
        
        Returns:
            None
        """
        # First get the fully connected layer indices
        first_network = self.networks[0]
        network_fc_indices, layers = self.find_fc_layers(first_network)
        
        # Get the bin ranges using those indices
        self.layer_bin_ranges = self.get_layer_bin_ranges(network_fc_indices)
        
        # Initialize weight distributions
        layer_weight_distributions_networks_skeleton = []
        for layer_idx, fc_idx in enumerate(network_fc_indices):
            layer = layers[fc_idx]
            layer_shape = layer.weight.shape
            
            # Create skeleton for this layer
            layer_skeleton = self.create_layer_skeleton(layer_shape)
            layer_weight_distributions_networks_skeleton.append(layer_skeleton)
        
        # Store weights from all networks
        network_weights = self.store_network_weights(network_fc_indices)

        # Populate bins with networks
        self.layer_weight_distributions_networks = self.populate_all_bins(network_weights, layer_weight_distributions_networks_skeleton)
        
        # Derive counts from networks
        self.layer_weight_distributions_counts = self.derive_counts_from_networks(self.layer_weight_distributions_networks)
        
    def create_conditional_weight_matrix(self, layer, given_weight_position, experimental_weight_position):
        
        # Assert that weight positions are tuples
        assert isinstance(given_weight_position, tuple), "given_weight_position must be a tuple (neuron_idx, from_weight_idx)"
        assert isinstance(experimental_weight_position, tuple), "experimental_weight_position must be a tuple (neuron_idx, from_weight_idx)"
        
        
        #get the networks which fall into the given bin
        given_distribution_networks = self.layer_weight_distributions_networks[layer][given_weight_position[0]][given_weight_position[1]][:]

        conditional_counts_matrix = np.zeros((self.num_bins, self.num_bins))
        #for each bin in the given distribution, bin the set of networks which fall into that bin
        for given_bin_idx, given_bin in enumerate(given_distribution_networks):
            if len(given_bin) == 0:
                # No networks in this bin, create empty distribution
                continue
            else:
                # layer_shape = given_bin[0].shape
                # skeleton = self.create_layer_skeleton(layer_shape)
                # binned_networks_bin_subsest = self.populate_bins_single_layer(layer, skeleton, network, self.layer_bin_ranges[layer])
                experimental_binned_networks = self.bin_single_weight(experimental_weight_position, given_bin, self.layer_bin_ranges[layer])
                for experimental_bin_idx, experimental_bin in enumerate(experimental_binned_networks):
                    conditional_counts_matrix[given_bin_idx][experimental_bin_idx] = len(experimental_bin)
        return conditional_counts_matrix

    def plot_heatmap(self, matrix_data, bin_edges, layer, position1, position2, title, colorbar_label, filename_prefix, cmap='viridis'):
        """
        Helper method to plot a heatmap with bin edges for weight analysis.
        
        Parameters:
        - matrix_data: 2D array of values to plot (could be probabilities, counts, etc.)
        - bin_edges: Array of bin edge values
        - layer: Index of the layer
        - position1: Tuple (neuron_idx, from_weight_idx) for x-axis
        - position2: Tuple (neuron_idx, from_weight_idx) for y-axis
        - title: Title for the plot
        - colorbar_label: Label for the colorbar
        - filename_prefix: Prefix for the saved file
        - cmap: Colormap to use (default: 'viridis')
        
        Returns:
        - fig, ax: The matplotlib figure and axis objects
        """
        # Create the heatmap figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Assert that bin edges are evenly spaced
        bin_widths = np.diff(bin_edges)
        avg_width = np.mean(bin_widths)
        max_deviation = np.max(np.abs(bin_widths - avg_width))
        tolerance = 1e-10  # Numeric tolerance for floating point comparison
        assert max_deviation < tolerance, f"Bin edges are not evenly spaced. Max deviation: {max_deviation}"
        
        # For use with shading='flat', match dimensions properly
        X, Y = np.meshgrid(bin_edges, bin_edges)
        im = ax.pcolormesh(X, Y, matrix_data, cmap=cmap, shading='flat')
        
        # Set grid lines to align with bin edges
        ax.set_xticks(bin_edges)
        ax.set_yticks(bin_edges)
        ax.set_xticklabels([f"{val:.2f}" for val in bin_edges], rotation=45)
        ax.set_yticklabels([f"{val:.2f}" for val in bin_edges])
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)
        
        # Configure axis labels and title
        ax.set_xlabel(f'Weight at position {position1}')
        ax.set_ylabel(f'Weight at position {position2}')
        ax.set_title(title)
        
        # Add grid lines at every bin edge
        ax.grid(True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{filename_prefix}_layer{layer}_{position1}_{position2}.png")
        
        return fig, ax

    def plot_conditional_weight_matrix(self, layer, given_weight_position, experimental_weight_position):
        """
        Plot a heatmap of the conditional weight distributions between two weights.
        
        This function calculates how the distribution of one weight depends on the value
        of another weight by analyzing the conditional distribution across networks.
        
        Parameters:
        - layer: Index of the layer (0-indexed)
        - given_weight_position: Tuple (neuron_idx, from_weight_idx) for the conditioning weight
        - experimental_weight_position: Tuple (neuron_idx, from_weight_idx) for the weight to predict
        
        Returns:
        - fig, ax: The matplotlib figure and axis objects
        
        Notes:
        - The heatmap rows (y-axis) represent bins of the given (conditioning) weight
        - The heatmap columns (x-axis) represent bins of the experimental (predicted) weight
        - Each cell shows how many networks have the experimental weight in a certain bin
          given that the conditioning weight is in a specific bin
        """
        # Get the conditional counts matrix
        conditional_matrix = self.create_conditional_weight_matrix(
            layer, given_weight_position, experimental_weight_position
        )
        
        # Get bin edges for this layer
        bin_edges = self.layer_bin_ranges[layer]
        
        # Create descriptive title and labels
        title = f"Conditional Distribution P(w{experimental_weight_position} | w{given_weight_position})"
        colorbar_label = "Count"
        filename_prefix = f"conditional_weight_matrix_layer{layer}"
        
        # Plot the heatmap using the plotting method
        return self.plot_heatmap(
            matrix_data=conditional_matrix,
            bin_edges=bin_edges,
            layer=layer,
            position1=given_weight_position,
            position2=experimental_weight_position,
            title=title,
            colorbar_label=colorbar_label,
            filename_prefix=filename_prefix,
            cmap='plasma'  # Using plasma colormap for conditional counts
        )

    def plot_weight_bins(self, weight_distributions, layer, weight_position, bin_edges):
            """
            Parameters:
            - weight_distributions: weight counts in the shape (layer, neuron, incoming weight, bin)
            - layer: Index of the layer to plot (0-indexed).
            - weight_position: Tuple indicating the neuron index and incoming weight index in the layer.
            - bin_edges: The edges of the bins for the specified layer, used to label the x-axis with actual values.
            """
            # get the bin data for the specified weight
            weight_distribution = weight_distributions[layer][weight_position[0]][weight_position[1]][:]

            num_bins = len(weight_distribution)

            bin_labels = [f"{bin_edges[i]:.3f} - {bin_edges[i+1]:.3f}" for i in range(num_bins)]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(num_bins), weight_distribution, width=0.8, align='center', edgecolor='black')
            plt.xlabel('Bin Range')
            plt.ylabel('Count')
            plt.title(f'Weight Histogram for Layer {layer}, Position {weight_position}')
            plt.xticks(range(num_bins), bin_labels, rotation=45, ha='right')  # Rotate for readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()  # Adjust layout to prevent label cut-off
            plt.show()
            plt.savefig(self.save_dir+ "/weight_plot_"+"layer:"+ str(layer) +"_position:"+str(weight_position))

    def normalize_distributions(self):
        """
        Normalize the weight distributions so that each bin represents a probability.
        Creates self.normalized_distributions from self.layer_weight_distributions_counts.
        """
        self.normalized_distributions = []
        for layer in self.layer_weight_distributions_counts:
            layer_counts = []
            for neuron in layer:
                neuron_counts = []
                for from_weight in neuron:
                    #total should be num of networks
                    total = sum(from_weight)
                    normalized_bin_counts = [bin_count / total for bin_count in from_weight]
                    neuron_counts.append(np.array(normalized_bin_counts))
                layer_counts.append(np.array(neuron_counts))
            self.normalized_distributions.append(np.array(layer_counts))

        return self.normalized_distributions

    def fit_distributions(self, multi_peak=False, peaks_per_layer=None, random_state=None, replicate_factor=1000):
        """
        Fit distributions to weight histograms.
        
        Parameters:
        - multi_peak: If True, fit GMMs; if False, fit single Gaussians (default False)
        - peaks_per_layer: List specifying components per layer (required if multi_peak=True)
        - random_state: Random seed for reproducible GMM fitting
        - replicate_factor: Multiplier for normalized counts in GMM fitting
        
        Sets:
        - self.fit_params (single Gaussians) or self.gmm_models (multi-peak GMMs)
        """
        # Ensure distributions are normalized
        if not hasattr(self, 'normalized_distributions'):
            self.normalize_distributions()
            
        if multi_peak:
            # GMM implementation
            if peaks_per_layer is None:
                raise ValueError("peaks_per_layer must be provided when multi_peak=True")
                
            self.gmm_models = []

            for layer_idx, (layer_data, bin_edges) in enumerate(zip(self.normalized_distributions, self.layer_bin_ranges)):
                n_components = peaks_per_layer[layer_idx]
                num_neurons, num_from_weights, num_bins = layer_data.shape
                layer_models = np.empty((num_neurons, num_from_weights), dtype=object)

                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                for neuron_idx in range(num_neurons):
                    for from_weight_idx in range(num_from_weights):
                        counts = layer_data[neuron_idx, from_weight_idx, :]
                        if np.any(counts):
                            X_list = []
                            for center, c in zip(bin_centers, counts):
                                #scaling up normalized count
                                if c < 1:
                                    rep_count = int(round(c * replicate_factor))
                                else:
                                    rep_count = int(c)
                                    
                                if rep_count > 0:
                                    X_list.extend([center] * rep_count)

                            if len(X_list) <= 1:  # If we have 0 or 1 sample
                                print(f"WARNING: Insufficient samples for GMM at layer {layer_idx}, neuron {neuron_idx}, input {from_weight_idx}. Adding noise to create multiple samples.")
                                # Find the bin with maximum count
                                max_bin_idx = np.argmax(counts)
                                # Create at least 2 samples with tiny random variations
                                X = np.array([bin_centers[max_bin_idx]] * 3).reshape(-1, 1)
                                # Add tiny variations to create distinct points
                                X[1] += 1e-6
                                X[2] -= 1e-6
                            else:
                                X = np.array(X_list).reshape(-1, 1)
                                
                            gmm = GaussianMixture(
                                n_components=n_components,
                                random_state=random_state
                            )
                            gmm.fit(X)
                            layer_models[neuron_idx, from_weight_idx] = gmm
                        else:
                            layer_models[neuron_idx, from_weight_idx] = None

                self.gmm_models.append(layer_models)
        else:
            # Single Gaussian implementation
            self.fit_params = []
            for layer_idx, (layer_data, bin_edges) in enumerate(zip(self.normalized_distributions, self.layer_bin_ranges)):
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                num_neurons, num_from_weights, _ = layer_data.shape
                layer_fit_params = np.zeros((num_neurons, num_from_weights, 3))

                for neuron_idx in range(num_neurons):
                    for from_weight_idx in range(num_from_weights):
                        weight_distribution = layer_data[neuron_idx, from_weight_idx, :]
                        #check for existence
                        if np.any(weight_distribution):
                            initial_guess = [
                                np.max(weight_distribution),
                                bin_centers[np.argmax(weight_distribution)],
                                np.std(bin_centers)
                            ]
                            try:
                                #fitting
                                popt, _ = curve_fit(self.gaussian, bin_centers, weight_distribution, p0=initial_guess)
                            except RuntimeError:
                                popt = initial_guess
                        else:
                            popt = [0, 0, 0]
                        layer_fit_params[neuron_idx, from_weight_idx, :] = popt
                self.fit_params.append(layer_fit_params)

    def gaussian(self, x, A, mu, sigma):
        """Gaussian function with amplitude, mean, and standard deviation"""
        return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

    def compute_kl_divergence(self, model1, model2, multi_peak=False, n_samples=10000):
        """
        Compute KL divergence between two models (Gaussians or GMMs).
        
        Parameters:
        - model1, model2: Models to compare 
          (For single-peak: (mu, sigma) tuples, For multi-peak: GMM objects)
        - multi_peak: If True, treat models as GMMs; if False, as Gaussians
        - n_samples: Number of samples for Monte Carlo estimation (multi_peak only)
        
        Returns:
        - KL divergence value
        """
        if multi_peak:
            # GMM KL divergence implementation
            gmm1, gmm2 = model1, model2
            if gmm1 is None or gmm2 is None:
                return float('inf')
            
            # Generate samples from gmm1
            X_samples, _ = gmm1.sample(n_samples)
            
            # Compute log probabilities under both models
            log_prob_gmm1 = gmm1.score_samples(X_samples)
            log_prob_gmm2 = gmm2.score_samples(X_samples)
            
            # KL divergence is E_p[log(p/q)] = E_p[log p - log q]
            kl_divergence = np.mean(log_prob_gmm1 - log_prob_gmm2)
            
            return max(0, kl_divergence)  # Ensure non-negative
        else:
            # Single Gaussian KL divergence implementation
            mu1, sigma1 = model1
            mu2, sigma2 = model2
            sigma1_adj = max(sigma1, 1e-8)
            sigma2_adj = max(sigma2, 1e-8)
            kl_divergence = np.log(sigma2_adj / sigma1_adj) + \
                (sigma1_adj ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2_adj ** 2) - 0.5
            return kl_divergence

    def cluster_distributions(self, threshold=0.001, multi_peak=False):
        """
        Cluster distributions within each layer based on KL divergence.
        
        Parameters:
        - threshold: KL divergence threshold for clustering
        - multi_peak: If True, cluster GMMs; if False, cluster Gaussians
        
        Returns:
        - cluster_indices_list: List of cluster indices for each layer
        """
        if multi_peak:
            # GMM clustering implementation
            if not hasattr(self, 'gmm_models'):
                print("No GMM models found. Call fit_distributions(multi_peak=True) first.")
                return None
            
            models = self.gmm_models
            cluster_indices_list = []
            
            for layer_idx, layer_models in enumerate(models):
                num_neurons, num_from_weights = layer_models.shape
                total_models = num_neurons * num_from_weights
                
                # Flatten models
                flattened_models = layer_models.flatten()
                
                # Calculate pairwise KL divergence
                kl_matrix = np.zeros((total_models, total_models))
                for idx1 in range(total_models):
                    for idx2 in range(idx1 + 1, total_models):
                        # Calculate symmetrized KL divergence
                        kl_forward = self.compute_kl_divergence(
                            flattened_models[idx1], flattened_models[idx2], multi_peak=True)
                        kl_backward = self.compute_kl_divergence(
                            flattened_models[idx2], flattened_models[idx1], multi_peak=True)
                        kl = (kl_forward + kl_backward) / 2  # Symmetrized version
                        
                        kl_matrix[idx1, idx2] = kl
                        kl_matrix[idx2, idx1] = kl
                
                # Perform clustering
                adjacency_matrix = kl_matrix < threshold
                cluster_indices = self.perform_clustering(adjacency_matrix, total_models)
                
                # Reshape to match layer structure
                cluster_indices_layer = cluster_indices.reshape((num_neurons, num_from_weights))
                cluster_indices_list.append(cluster_indices_layer)
            
            return cluster_indices_list
            
        else:
            # Single Gaussian clustering implementation
            if not hasattr(self, 'fit_params'):
                print("No Gaussian fits found. Call fit_distributions(multi_peak=False) first.")
                return None
                
            fit_params = self.fit_params
            cluster_indices_list = []

            for layer_idx, layer_params in enumerate(fit_params):
                num_neurons, num_from_weights, _ = layer_params.shape
                total_models = num_neurons * num_from_weights

                # Flatten parameters
                mu_sigma_list = []
                for i in range(num_neurons):
                    for j in range(num_from_weights):
                        A, mu, sigma = layer_params[i, j, :]
                        mu_sigma_list.append((mu, sigma))

                # Calculate pairwise KL divergence
                kl_matrix = np.zeros((total_models, total_models))
                for idx1 in range(total_models):
                    for idx2 in range(idx1 + 1, total_models):
                        kl = self.compute_kl_divergence(
                            mu_sigma_list[idx1], mu_sigma_list[idx2], multi_peak=False)
                        kl_matrix[idx1, idx2] = kl
                        kl_matrix[idx2, idx1] = kl 

                # Perform clustering
                adjacency_matrix = kl_matrix < threshold
                cluster_indices = self.perform_clustering(adjacency_matrix, total_models)
                
                # Reshape to match layer structure
                cluster_indices_layer = cluster_indices.reshape((num_neurons, num_from_weights))
                cluster_indices_list.append(cluster_indices_layer)

            return cluster_indices_list

    def perform_clustering(self, adjacency_matrix, total_elements):
        """
        Helper method to perform clustering given an adjacency matrix.
        
        Parameters:
        - adjacency_matrix: Boolean matrix where True indicates elements should be in the same cluster
        - total_elements: Total number of elements to cluster
        
        Returns:
        - cluster_indices: Array indicating cluster assignment for each element
        """
        cluster_indices = np.full(total_elements, -1, dtype=int)
        current_cluster = 0
        
        for idx in range(total_elements):
            if cluster_indices[idx] == -1:
                cluster_indices[idx] = current_cluster
                queue = [idx]
                while queue:
                    current_idx = queue.pop(0)
                    neighbors = np.where(adjacency_matrix[current_idx])[0]
                    for neighbor_idx in neighbors:
                        if cluster_indices[neighbor_idx] == -1:
                            cluster_indices[neighbor_idx] = current_cluster
                            queue.append(neighbor_idx)
                current_cluster += 1
                
        return cluster_indices

    def plot_distribution(self, layer, weight_position, multi_peak=False, scale_to_hist=True, show_individual=True):
        """
        Plot distribution for a specific weight position.
        
        Parameters:
        - layer: Layer index
        - weight_position: (neuron_idx, from_weight_idx) tuple
        - multi_peak: If True, plot GMM; if False, plot single Gaussian
        - scale_to_hist: Whether to scale the PDF to match histogram height
        - show_individual: For multi_peak, whether to show individual components
        """
        neuron_idx, from_weight_idx = weight_position
        weight_distribution = self.normalized_distributions[layer][neuron_idx, from_weight_idx, :]
        bin_edges = self.layer_bin_ranges[layer]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, weight_distribution, width=np.diff(bin_edges),
                align='center', edgecolor='black', alpha=0.6, label='Histogram')

        if multi_peak:
            # GMM plotting implementation
            gmm = self.gmm_models[layer][neuron_idx, from_weight_idx]
            if gmm is None:
                print("No data or GMM not fitted for this distribution.")
                return

            # Evaluate the GMM PDF on a smooth grid
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000).reshape(-1, 1)
            log_probs = gmm.score_samples(x_fit)
            pdf = np.exp(log_probs)

            # Scale the fit to match the height of histogram, optional
            counts_sum = np.sum(weight_distribution)
            bin_width = (bin_edges[-1] - bin_edges[0]) / (len(bin_edges) - 1) if len(bin_edges) > 1 else 1.0
            if scale_to_hist:
                pdf *= (counts_sum * bin_width)

            plt.plot(x_fit, pdf, 'r-', label='GMM PDF')

            # Plot each component
            if show_individual:
                for i, (w, m, c) in enumerate(zip(gmm.weights_, gmm.means_, gmm.covariances_)):
                    mu = m[0]
                    sigma = np.sqrt(c[0, 0])
                    component_pdf = w * norm.pdf(x_fit.flatten(), mu, sigma)
                    if scale_to_hist:
                        component_pdf *= (counts_sum * bin_width)
                    plt.plot(x_fit, component_pdf, '--', label=f'Component {i+1}')

            title_suffix = "GMM Fit"
        else:
            # Single Gaussian plotting implementation
            if not hasattr(self, 'fit_params'):
                print("No Gaussian fits found. Call fit_distributions(multi_peak=False) first.")
                return
                
            A, mu, sigma = self.fit_params[layer][neuron_idx, from_weight_idx, :]
            
            # Generate fitted curve
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
            y_fit = self.gaussian(x_fit, A, mu, sigma)
            
            plt.plot(x_fit, y_fit, 'r-', label='Gaussian Fit')
            
            title_suffix = "Gaussian Fit"

        plt.xlabel('Weight Value')
        plt.ylabel('Probability')
        plt.title(f'Layer {layer}, Position {weight_position} - {title_suffix}')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Save with appropriate filename
        filename = f"/{'gmm' if multi_peak else 'gaussian'}_plot_layer{layer}_pos{weight_position}"
        plt.savefig(self.save_dir + filename)

    def plot_unique_distributions(self, indices, layer, multi_peak=False):
        """
        Plot unique distributions for a specified layer based on cluster indices.
        
        Parameters:
        - indices: List of cluster indices from cluster_distributions
        - layer: The layer index to plot unique distributions for
        - multi_peak: If True, plot GMMs; if False, plot single Gaussians
        """
        # Check if the required models exist
        if multi_peak and not hasattr(self, 'gmm_models'):
            print("No GMM models found. Call fit_distributions(multi_peak=True) first.")
            return
        elif not multi_peak and not hasattr(self, 'fit_params'):
            print("No Gaussian fits found. Call fit_distributions(multi_peak=False) first.")
            return
        
        layer_indices = indices[layer]
        
        seen_cluster_indices = set()
        for i in range(layer_indices.shape[0]):
            for j in range(layer_indices.shape[1]):
                cluster_idx = layer_indices[i, j]
                
                # If we haven't seen this cluster index before, plot it
                if cluster_idx not in seen_cluster_indices:
                    print(f"Plotting unique {'GMM' if multi_peak else 'Gaussian'} cluster {cluster_idx} at position ({i}, {j})")
                    self.plot_distribution(
                        layer=layer,
                        weight_position=(i, j),
                        multi_peak=multi_peak,
                        scale_to_hist=True,
                        show_individual=True
                    )
                    seen_cluster_indices.add(cluster_idx)
                
        print(f"Found {len(seen_cluster_indices)} unique {'GMM' if multi_peak else 'Gaussian'} distributions in layer {layer}")

    def weights_walkthrough(self, layer=0, walkthrough_strategy='previous_layer_first', start_neuron=0):
        """
        Step through the weights of a selected layer using a specified traversal strategy.

        Parameters:
        - layer: Index of the selected layer (0-indexed).
        - walkthrough_strategy: 'previous_layer_first' or 'next_layer_first' traversal strategy.
        - start_neuron: Optional, neuron in the selected layer to start the walkthrough from (default is 0).
        """
        if not hasattr(self, 'layer_weight_distributions_counts') or not hasattr(self, 'layer_bin_ranges'):
            print("No weight distributions available. Call store_weights() first.")
            return
        
        weight_distributions = self.layer_weight_distributions_counts
        bin_edges = self.layer_bin_ranges[layer]
        
        num_neurons = weight_distributions[layer].shape[0]  # number of neurons in the selected layer
        num_weights = weight_distributions[layer].shape[1]  # number of weights to each neuron in selected layer

        if walkthrough_strategy == 'previous_layer_first':
            for neuron in range(start_neuron, num_neurons):
                for incoming_weight in range(num_weights):
                    print(f"Neuron {neuron}, Incoming Weight {incoming_weight}")
                    
                    self.plot_weight_bins(weight_distributions, layer, (neuron, incoming_weight), bin_edges)
                    
                    user_input = input("Press Enter to continue, or type 'q' to stop: ").strip().lower()
                    if user_input == 'q':
                        print("Quitting walkthrough.")
                        return
                
        elif walkthrough_strategy == 'next_layer_first':
            for incoming_weight in range(num_weights):
                for neuron in range(start_neuron, num_neurons):
                    print(f"Neuron {neuron}, Incoming Weight {incoming_weight}")
                    
                    self.plot_weight_bins(weight_distributions, layer, (neuron, incoming_weight), bin_edges)
                    
                    user_input = input("Press Enter to continue, or type 'q' to stop: ").strip().lower()
                    if user_input == 'q':
                        print("Quitting walkthrough.")
                        return
                
        else:
            print("Invalid walkthrough strategy. Choose 'previous_layer_first' or 'next_layer_first'.")

    def analyze_distributions(self, layer=0, kl_threshold=0.001, multi_peak=False, peaks_per_layer=None, 
                             random_state=42, replicate_factor=1000, run_walkthrough=False, 
                             walkthrough_strategy='previous_layer_first', start_neuron=0):
        """
        Driver function to analyze weight distributions in a specified layer.
        
        Parameters:
        - layer: Layer index to analyze (default 0)
        - kl_threshold: Threshold for KL divergence clustering (default 0.001)
        - multi_peak: Whether to use multi-peak GMMs instead of single Gaussians (default False)
        - peaks_per_layer: List specifying number of components for each layer's GMM (required if multi_peak=True)
        - random_state: Random seed for GMM fitting (default 42)
        - replicate_factor: Multiplier for normalized counts in GMM fitting (default 1000)
        - run_walkthrough: Whether to run an interactive walkthrough of weight distributions (default False)
        - walkthrough_strategy: Strategy for weight walkthrough, 'previous_layer_first' or 'next_layer_first' (default 'previous_layer_first')
        - start_neuron: Starting neuron for walkthrough (default 0)
        
        Returns:
        - cluster_indices: List of cluster indices for each layer
        """

        # Run interactive walkthrough if requested
        if run_walkthrough:
            print("Starting interactive weight walkthrough...")
            self.weights_walkthrough(
                layer=layer,
                walkthrough_strategy=walkthrough_strategy,
                start_neuron=start_neuron
            )

        # Ensure distributions are normalized
        if not hasattr(self, 'normalized_distributions'):
            print("Normalizing distributions...")
            self.normalize_distributions()
        
        # Fit distributions
        print(f"Fitting {'GMM' if multi_peak else 'Gaussian'} models...")
        self.fit_distributions(
            multi_peak=multi_peak,
            peaks_per_layer=peaks_per_layer,
            random_state=random_state,
            replicate_factor=replicate_factor
        )
        
        # Cluster distributions
        print(f"Clustering {'GMMs' if multi_peak else 'Gaussians'} with KL threshold {kl_threshold}...")
        cluster_indices = self.cluster_distributions(
            threshold=kl_threshold,
            multi_peak=multi_peak
        )
        
        # Plot unique distributions
        print(f"Plotting unique {'GMM' if multi_peak else 'Gaussian'} distributions for layer {layer}...")
        self.plot_unique_distributions(
            indices=cluster_indices,
            layer=layer,
            multi_peak=multi_peak
        )        
        
        return cluster_indices

    def find_fc_layers(self, network):
        """
        Find the indices of fully connected (Linear) layers in a network.
        
        Parameters:
        - network: A PyTorch neural network
        
        Returns:
        - fc_indices: A list of indices where nn.Linear layers are found
        - layers: The list of all layers in the network
        """
        # Get the appropriate layers from the network
        if hasattr(network, "network"):
            layers = list(network.network.children())
        else:
            layers = list(network.children())
        
        # Find indices of Linear layers
        fc_indices = []
        for index, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                fc_indices.append(index)
        
        return fc_indices, layers
        
    def get_layer_bin_ranges(self, fc_indices):
        """
        Calculate bin ranges for each fully connected layer based on min/max weight values.
        
        Parameters:
        - fc_indices: List of indices corresponding to fully connected layers
        
        Returns:
        - layer_bin_ranges: A list of bin edges for each fully connected layer
        """
        if not hasattr(self, 'networks') or len(self.networks) == 0:
            raise ValueError("Networks not loaded. Call load_models() first.")
            
        # Initialize layer_bin_ranges
        layer_bin_ranges = []
        
        # Calculate bin ranges for each FC layer
        for fc_idx, _ in enumerate(fc_indices):
            layerMin = self.__getMinOrMax__(self.networks, fc_idx, True)
            layerMax = self.__getMinOrMax__(self.networks, fc_idx, False)
            
            # Calculate bin edges
            bin_edges = np.histogram_bin_edges(a=[], bins=self.num_bins, range=(layerMin, layerMax))
            layer_bin_ranges.append(bin_edges)
        
        return layer_bin_ranges

    