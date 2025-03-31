#!/usr/bin/env python
"""
This script uses multiprocessing to concurrently train neural networks on MNIST.
Each worker is assigned to one GPU to train a network.
Only networks that meet the success criteria (test accuracy >= success_accuracy
and training loss convergence) are saved.

Usage:
    python MNIST_testing.py [--subset] [--amount NUM] [--output PATH]
    
    --subset: Optional flag to use only digits 1-8 instead of all digits 0-9
    --amount: Number of successful networks to produce (default: 2000)
    --output: Output file path and name (default: WorkingNetworks/working_networks_mnist.pt)
              A suffix will be added if --subset is used

Examples:
    # Train 100 networks on all digits, save to default location
    python MNIST_testing.py --amount 100
    
    # Train 50 networks on digits 1-8, save to custom location
    python MNIST_testing.py --subset --amount 50 --output my_results/mnist_networks.pt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from concurrent.futures import ProcessPoolExecutor
from itertools import cycle
import torch.multiprocessing as mp
import json
from datetime import datetime
import argparse

def load_data(batch_size=64, data_dir="./MNIST_data", digits_subset=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Note: download=False because data is pre-downloaded
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    
    # Filter dataset to include only specified digits if digits_subset is provided
    if digits_subset is not None:
        # Create a mapping from original labels to new consecutive labels (0 to len(digits_subset)-1)
        label_mapping = {label: i for i, label in enumerate(sorted(digits_subset))}
        
        # Filter and remap training data
        train_indices = []
        train_targets = []
        
        # Get all targets as a list for easier processing
        all_train_targets = train_dataset.targets.tolist()
        
        for i, label in enumerate(all_train_targets):
            if label in digits_subset:
                train_indices.append(i)
                # Store the remapped label
                train_targets.append(label_mapping[label])
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        # Override the targets with remapped values
        train_dataset.dataset.targets[train_indices] = torch.tensor(train_targets)
        
        # Filter and remap test data
        test_indices = []
        test_targets = []
        
        # Get all targets as a list for easier processing
        all_test_targets = test_dataset.targets.tolist()
        
        for i, label in enumerate(all_test_targets):
            if label in digits_subset:
                test_indices.append(i)
                # Store the remapped label
                test_targets.append(label_mapping[label])
        
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        # Override the targets with remapped values
        test_dataset.dataset.targets[test_indices] = torch.tensor(test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

class MinimalMNIST(nn.Module):
    def __init__(self):
        super(MinimalMNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 10)  # 10 output classes for MNIST
       
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# Define the custom model class at module level, not inside if __name__ == '__main__'
class MinimalMNISTCustom(nn.Module):
    def __init__(self, output_classes=10):
        super(MinimalMNISTCustom, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, output_classes)  # Configurable output classes

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

def create_model(output_classes=10):
    """Create model with specified number of output classes - defined at module level"""
    return MinimalMNISTCustom(output_classes=output_classes)

def train_single_network(output_classes, train_loader, test_loader, num_epochs,
                         learning_rate, loss_fn, device, success_accuracy, convergence_threshold, task_id):
    # Create the model directly here instead of passing a factory function
    model = MinimalMNISTCustom(output_classes=output_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        # print(f"Task {task_id} Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Check for convergence based on loss improvement
    if len(train_loss_history) >= 2:
        converged = (train_loss_history[-2] - train_loss_history[-1]) < convergence_threshold
    else:
        converged = False

    # Evaluate test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f"Task {task_id} Test Accuracy: {accuracy:.4f}")
    
    # Check for success: test accuracy must be above threshold and training loss must have converged
    if accuracy >= success_accuracy and converged:
        print(f"Network {task_id} succeeded with accuracy {accuracy:.4f}.")
        return model.state_dict()
    else:
        print(f"Network {task_id} failed with accuracy {accuracy:.4f}.")
        print(f"Network {task_id} failed with convergence {(train_loss_history[-2] - train_loss_history[-1])}.")
        return None

def worker(task_id, gpu_id, output_classes, train_loader, test_loader,
           num_epochs, learning_rate, loss_fn, success_accuracy, convergence_threshold):
    # Set the visible GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = train_single_network(output_classes, train_loader, test_loader,
                               num_epochs, learning_rate, loss_fn, device,
                               success_accuracy, convergence_threshold, task_id)
    return result

def parallel_train_networks(output_classes, train_loader, test_loader,
                            num_epochs, learning_rate, loss_fn,
                            success_accuracy, convergence_threshold,
                            amount_to_produce, max_workers=4):
    collected_networks = {}
    tasks_submitted = 0
    tasks_successful = 0

    # Create a pool of workers (one per GPU)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        gpu_cycle = cycle(range(max_workers))
        while tasks_successful < amount_to_produce:
            gpu_id = next(gpu_cycle)
            future = executor.submit(worker, tasks_submitted, gpu_id, output_classes,
                                     train_loader, test_loader, num_epochs,
                                     learning_rate, loss_fn, success_accuracy,
                                     convergence_threshold)
            futures.append((tasks_submitted, future))
            tasks_submitted += 1

            for task_id, fut in list(futures):
                if fut.done():
                    try:
                        result = fut.result()
                    except Exception as e:
                        print(f"Task {task_id} encountered an error: {e}")
                        result = None
                    if result is not None:
                        collected_networks[f'network_{tasks_successful+1}'] = result
                        tasks_successful += 1
                        print(f"Network {tasks_successful} succeeded (Task {task_id}).")
                    futures.remove((task_id, fut))
                    if tasks_successful >= amount_to_produce:
                        for _, pending_fut in futures:
                            pending_fut.cancel()
                        futures.clear()
                        break
    return collected_networks

# -----------------------------
# Main Script
# -----------------------------
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural networks on MNIST dataset')
    parser.add_argument('--subset', action='store_true', help='Use only digits 1-8 instead of all digits 0-9')
    parser.add_argument('--amount', type=int, default=2000, help='Number of successful networks to produce')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file path and name (default: WorkingNetworks/working_networks_mnist.pt)')
    args = parser.parse_args()
    
    # Record start time
    start_time = datetime.now()
    
    mp.set_start_method('spawn')
    
    # Choose which digits to use based on command line argument
    digits_subset = [1, 2, 3, 4, 5, 6, 7, 8] if args.subset else None
    
    # Load data from the pre-downloaded directory
    batch_scale = 16
    train_loader, test_loader = load_data(batch_size=(64 * 16), data_dir="./MNIST_data", digits_subset=digits_subset)
    
    # Use CrossEntropyLoss for classification
    loss_fn = nn.CrossEntropyLoss()
    
    # Adjust output layer size if using a subset of digits
    output_classes = len(digits_subset) if digits_subset is not None else 10
    
    # Set success criteria: target test accuracy and convergence threshold
    success_accuracy = 0.10  # require at least 90% test accuracy
    convergence_threshold = 1e-1  # threshold for improvement in training loss
    base_lr = 0.05
    amount_to_produce = args.amount  # number of successful networks to produce from command line
    num_epochs = 5
    learning_rate = (base_lr * (batch_scale ** 0.5))
    
    # Pass output_classes directly rather than a factory function
    networks_state_dicts = parallel_train_networks(
        output_classes,  # Pass the number of output classes directly
        train_loader,
        test_loader,
        num_epochs,
        learning_rate,
        loss_fn,
        success_accuracy,
        convergence_threshold,
        amount_to_produce,
        max_workers=4  # assumes 4 GPUs available
    )

    # Determine output file path
    if args.output is None:
        # Create a descriptive suffix for the output file based on digits used
        digits_suffix = "_digits_1to8" if args.subset else ""
        output_file = f"WorkingNetworks/working_networks_mnist{digits_suffix}.pt"
    else:
        output_file = args.output
        # If using a custom output path, ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Ensure the default directory exists if using default output
    if args.output is None:
        os.makedirs("WorkingNetworks", exist_ok=True)
        
    torch.save(networks_state_dicts, output_file)
    
    # Record end time and create metadata
    end_time = datetime.now()
    duration = end_time - start_time
    
    metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "batch_size": 64 * batch_scale,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "networks_produced": amount_to_produce,
        "success_accuracy_threshold": success_accuracy,
        "digits_used": digits_subset if digits_subset is not None else "all (0-9)",
        "output_file": output_file
    }
    
    # Save metadata
    with open(output_file + "_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Training complete. Duration: {duration}")
    print(f"Saved {amount_to_produce} networks and metadata to {output_file}")
    print(f"Used digits: {'1-8' if args.subset else 'all (0-9)'}")