import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import sys

# Add parent directory to Python path to import from sibling directories
# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(script_dir)
# Add parent directory to path
sys.path.append(parent_dir)

# Import network architectures from the proper module paths
from Architectures.permutation_free_architecture import PermutationFreeNet

#2 hidden layers
#2 inputs, 2 hidden with 5 neurons each
class SimpleNet(nn.Module):
    # 2 because a, b
    input_size = 2
    # num of hidden neurons
    hidden_size = 5
    output_size = 1
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # Right now RELU but we can change later
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def create_model(architecture):
    """Create a model based on the specified architecture"""
    if architecture == 'permutation_free':
        return PermutationFreeNet(use_masked=True)
    else:  # 'simple'
        return SimpleNet()

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze neural network accuracy')
    parser.add_argument('--networks', type=str, default="WorkingNetworks/working_networks.pt",
                        help='Path to the saved networks file')
    parser.add_argument('--data', type=str, default='./data/simpleReg.csv',
                        help='Path to the data CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save output files')
    parser.add_argument('--architecture', type=str, choices=['simple', 'permutation_free'], default='simple',
                        help='Network architecture to use (simple or permutation_free)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data_path = args.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data = pd.read_csv(data_path)
    X = data[['a', 'b']].values.astype(np.float32)
    y_true = data['y'].values.astype(np.float32)

    # Convert to torch tensors (y_true reshaped to [N,1])
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y_true).unsqueeze(1)

    networks_file = args.networks
    if not os.path.exists(networks_file):
        raise FileNotFoundError(f"Networks file not found at {networks_file}")

    # Add weights_only=False to suppress the warning
    networks_state_dicts = torch.load(networks_file, map_location=torch.device('cpu'), weights_only=False)

    mse_list = []
    mae_list = []
    r2_list = []

    # Use the specified architecture
    architecture = args.architecture
    print(f"Using architecture: {'Permutation Free Network' if architecture == 'permutation_free' else 'Simple Network'}")
    
    print(f"Evaluating networks from {networks_file}:")
    for net_name, state_dict in networks_state_dicts.items():
        # Create a model with the specified architecture
        model = create_model(architecture)
        try:
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor).squeeze().numpy()  # predictions shape: (N,)
            mse = mean_squared_error(y_true, predictions)
            mae = mean_absolute_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)
            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)
        except Exception as e:
            print(f"Error loading {net_name}: {e}")
            print(f"This might indicate a mismatch between the specified architecture and the saved model.")
            continue

    if not mse_list:
        print("No networks were successfully evaluated. Check if the architecture matches the saved models.")
        return

    print("\nSummary Metrics Across Networks:")
    print(f"Average MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
    print(f"Average MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"Average R2:  {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

    # Create a grid over the input space using the min/max of a and b from the data
    a_min, a_max = data['a'].min(), data['a'].max()
    b_min, b_max = data['b'].min(), data['b'].max()
    a_range = np.linspace(a_min, a_max, 50)
    b_range = np.linspace(b_min, b_max, 50)
    A, B = np.meshgrid(a_range, b_range)
    grid_inputs = np.column_stack([A.ravel(), B.ravel()]).astype(np.float32)
    grid_tensor = torch.tensor(grid_inputs)

    # Use the first successful model for visualization
    first_key = list(networks_state_dicts.keys())[0]
    model = create_model(architecture)
    model.load_state_dict(networks_state_dicts[first_key])
    model.eval()
    with torch.no_grad():
        grid_predictions = model(grid_tensor).squeeze().numpy()
    grid_predictions = grid_predictions.reshape(A.shape)

    def target_function(a, b):
        return (1/5) * a**2 - (1/10) * b**3

    Z_true = target_function(A, B)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create proxy artists for the legend with specific colors instead of colormaps
    fake2Dline1 = plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.5)
    fake2Dline2 = plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.5)

    # Update the surface plots to use the same colors as the legend
    surf1 = ax.plot_surface(A, B, grid_predictions, color='blue', 
                           edgecolor='none', alpha=0.5, label='Network Prediction')
    surf2 = ax.plot_surface(A, B, Z_true, color='red', 
                           edgecolor='none', alpha=0.5, label='True Function')

    # Add title and labels
    ax.set_title('Network Prediction vs True Function')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('z')

    # Add legend
    ax.legend([fake2Dline1, fake2Dline2], ['Network Prediction', 'True Function'], 
             loc='upper left')

    plt.tight_layout()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get network filename for plot title
    network_filename = os.path.basename(networks_file)
    arch_suffix = "_permfree" if architecture == "permutation_free" else "_simple"
    output_filename = f'network_regression_fit_{network_filename.split(".")[0]}{arch_suffix}.png'
    output_path = os.path.join(args.output_dir, output_filename)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()