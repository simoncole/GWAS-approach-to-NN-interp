#!/usr/bin/env python
"""
This script uses multiprocessing to concurrently train neural networks on MNIST.
Each worker is assigned to one GPU to train a network.
Only networks that meet the success criteria (test accuracy >= success_accuracy
and training loss convergence) are saved.

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

def load_data(batch_size=64, data_dir="./MNIST_data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Note: download=False because data is pre-downloaded
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes for MNIST
       
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_single_network(model_architecture, train_loader, test_loader, num_epochs,
                         learning_rate, loss_fn, device, success_accuracy, convergence_threshold, task_id):
    model = model_architecture().to(device)
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
        print(f"Task {task_id} Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
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

def worker(task_id, gpu_id, model_architecture, train_loader, test_loader,
           num_epochs, learning_rate, loss_fn, success_accuracy, convergence_threshold):
    # Set the visible GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = train_single_network(model_architecture, train_loader, test_loader,
                                  num_epochs, learning_rate, loss_fn, device,
                                  success_accuracy, convergence_threshold, task_id)
    return result

def parallel_train_networks(model_architecture, train_loader, test_loader,
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
            future = executor.submit(worker, tasks_submitted, gpu_id, model_architecture,
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
    mp.set_start_method('spawn')
    # Load data from the pre-downloaded directory (data_dir must match that used in download_mnist.py)
    train_loader, test_loader = load_data(batch_size=64, data_dir="./MNIST_data")
    
    # Use CrossEntropyLoss for classification
    loss_fn = nn.CrossEntropyLoss()
    
    # Set success criteria: target test accuracy and convergence threshold
    success_accuracy = 0.90  # require at least 90% test accuracy
    convergence_threshold = 1e-1  # threshold for improvement in training loss
    
    amount_to_produce = 5  # number of successful networks to produce
    num_epochs = 5
    learning_rate = 0.05

    networks_state_dicts = parallel_train_networks(
        SimpleNet,
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

    os.makedirs("WorkingNetworks", exist_ok=True)
    torch.save(networks_state_dicts, "WorkingNetworks/working_networks_mnist.pt")
    print("Training complete. Saved all networks.")