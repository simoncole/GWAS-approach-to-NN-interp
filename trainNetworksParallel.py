#!/usr/bin/env python
"""
This script uses multiprocessing to concurrently train neural networks. 
Each worker is assigned to one GPU
to train a network. Only networks that meet the success criteria are saved.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import cycle
import torch.multiprocessing as mp

from largerNetArchitecture import LargerNet
from standardRegArchitecture import SimpleNet

def train_single_network(model_architecture, train_loader, test_loader, num_epochs,
                         learning_rate, loss_fn, device, success_loss, convergence_threshold, task_id):
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
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    #check for convergence
    if len(train_loss_history) >= 2:
        converged = (train_loss_history[-2] - train_loss_history[-1]) < convergence_threshold
    else:
        converged = False
    if not converged: print("Network did not converge.")
    if train_loss_history[-1] > success_loss: print("Network loss is too high.")

    #check for success
    if train_loss_history[-1] <= success_loss and converged:
        print(f"Network {task_id} succeeded with loss {train_loss_history[-1]:.4f}.")
        return model.state_dict()
    else:
        return None

def worker(task_id, gpu_id, model_architecture, train_loader, test_loader,
           num_epochs, learning_rate, loss_fn, success_loss, convergence_threshold):
    #set visible gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    # print(f"Task {task_id} running on GPU {gpu_id}")
    result = train_single_network(model_architecture, train_loader, test_loader,
                                  num_epochs, learning_rate, loss_fn, device,
                                  success_loss, convergence_threshold, task_id)
    return result

def parallel_train_networks(model_architecture, train_loader, test_loader,
                            num_epochs, learning_rate, loss_fn,
                            success_loss, convergence_threshold,
                            amount_to_produce, max_workers=4):
    collected_networks = {}
    tasks_submitted = 0
    tasks_successful = 0

    #create a pool of workers (one per gpu)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        gpu_cycle = cycle(range(max_workers))
        while tasks_successful < amount_to_produce:
            gpu_id = next(gpu_cycle)
            future = executor.submit(worker, tasks_submitted, gpu_id, model_architecture,
                                     train_loader, test_loader, num_epochs,
                                     learning_rate, loss_fn, success_loss,
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
    return collected_networks

def load_data():
    csv_data = pd.read_csv('data/simpleReg.csv')
    a = csv_data['a'].values.reshape(-1, 1)
    b = csv_data['b'].values.reshape(-1, 1)
    y = csv_data['y'].values.reshape(-1, 1)

    a_tensor = torch.tensor(a, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    X_tensor = torch.cat((a_tensor, b_tensor), dim=1)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

if __name__ == '__main__':
    mp.set_start_method('spawn')
    print(torch.cuda.is_available())
    train_loader, test_loader = load_data()
    print("Data loaded.")
    loss_fn = nn.MSELoss()
    success_loss = 0.1              # Networks should have final loss <= 0.1
    convergence_threshold = 1e-1      # Threshold for determining convergence
    amount_to_produce = 100          # Total number of good networks desired
    num_epochs = 5                   # Training epochs per network
    learning_rate = 0.1

    networks_state_dicts = parallel_train_networks(
        SimpleNet,
        train_loader,
        test_loader,
        num_epochs,
        learning_rate,
        loss_fn,
        success_loss,
        convergence_threshold,
        amount_to_produce,
        max_workers=4  #in this case working with 4 gpus so 4 workers
    )

    os.makedirs("WorkingNetworks", exist_ok=True)
    torch.save(networks_state_dicts, "WorkingNetworks/working_networks.pt")
    print("Training complete. Saved all networks.")