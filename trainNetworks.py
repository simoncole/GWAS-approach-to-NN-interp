# trainNetworks.py

import argparse
import os
from network_analyzer import NetworkAnalyzer
import torch
from largerNetArchitecture import LargerNet
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from largerNetArchitecture import LargerNet
from standardRegArchitercture import SimpleNet
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser(description="Train networks using the network generation class")
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory in which to save working networks, broken networks, and figures.'
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    working_networks_path = os.path.join(output_dir, "WorkingNetworks")
    broken_networks_path = os.path.join(output_dir, "BrokenNetworks")
    figures_path = os.path.join(output_dir, "figures")

    os.makedirs(working_networks_path, exist_ok=True)
    os.makedirs(broken_networks_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    csv_data = pd.read_csv('data/simpleReg.csv')
    a = csv_data['a'].values.reshape(-1, 1)
    b = csv_data['b'].values.reshape(-1, 1)
    y = csv_data['y'].values.reshape(-1, 1)

    a_tensor = torch.tensor(a, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    X_tensor = torch.cat((a_tensor, b_tensor), dim=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    loss_fn = MSELoss()

    analyzer = NetworkAnalyzer(
        model_architecture=LargerNet,
        amount_to_produce=1000,
        success_loss=0.1,
        convergence_threshold=1,
        max_attempts=10
    )
    analyzer.working_dir = working_networks_path  
    analyzer.broken_dir = broken_networks_path    

    analyzer.generate_networks(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=10,
        loss_fn=loss_fn,
        learning_rate=0.1
    )

    loaded_state_dict = torch.load(os.path.join(working_networks_path, "working_networks.pt"))

    num_networks = len(loaded_state_dict)
    networks = []
    for n in range(num_networks):
        networks.append(SimpleNet())

    for i, network in enumerate(networks):
        state_dict_key = f'network_{i+1}'
        network.load_state_dict(loaded_state_dict[state_dict_key])

    def target_function(a, b):
        return (1/5) * a**2 - (1/10) * b**3

    network = networks[3]

    a_range = np.linspace(a_tensor.min(), a_tensor.max(), 50)
    b_range = np.linspace(b_tensor.min(), b_tensor.max(), 50)
    A, B = np.meshgrid(a_range, b_range)
    grid = np.c_[A.ravel(), B.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    network.eval()
    with torch.no_grad():
        y_pred_grid = network(grid_tensor).numpy()

    Y_pred_grid = y_pred_grid.reshape(A.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, Y_pred_grid, cmap='viridis', alpha=0.8)

    Y_actual_grid = target_function(A, B)
    ax.plot_wireframe(A, B, Y_actual_grid, color='r', label='Actual Function')

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('Predicted Y')
    plt.title('3D Surface Plot of Predicted Y over a and b')

    figure_filename = os.path.join(figures_path, "3D_surface_plot.png")
    plt.savefig(figure_filename)
    plt.close(fig)


if __name__ == '__main__':
    main()