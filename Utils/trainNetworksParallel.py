#!/usr/bin/env python
"""
This script uses multiprocessing to concurrently train neural networks with permutation-free architecture. 
Each worker is assigned to one GPU to train a network. Only networks that meet the success criteria are saved.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to Python path to import from sibling directories
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import cycle
import torch.multiprocessing as mp
import argparse
import json
from datetime import datetime
import gc
from concurrent.futures import as_completed

# Now import from sibling directory
from Architectures.permutation_free_architecture import PermutationFreeNet

def create_permutation_free_net(use_masked, itype, freeze):
    """Create a permutation-free network with the specified settings."""
    return PermutationFreeNet(
        use_masked=use_masked,
        itype=itype,
        freeze=freeze
    )

def train_single_network(model_architecture, permutation_free_settings, train_loader, test_loader, num_epochs,
                         learning_rate, loss_fn, device, success_loss, convergence_threshold, task_id):
    model = None
    optimizer = None
    try:
        model = model_architecture(**permutation_free_settings).to(device)
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
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        #check for convergence
        if len(train_loss_history) >= 2:
            converged = (train_loss_history[-2] - train_loss_history[-1]) < convergence_threshold
        else:
            converged = False

        if not converged: 
            print(f"Task {task_id}: Network did not converge.")
        if train_loss_history[-1] > success_loss: 
            print(f"Task {task_id}: Network loss is too high: {train_loss_history[-1]:.4f}")

        #check for success
        if train_loss_history[-1] <= success_loss and converged:
            print(f"Task {task_id}: Network succeeded with loss {train_loss_history[-1]:.4f}.")
            # Move the model to CPU first
            model.cpu()
            # Then get the state dict (which will now be on CPU)
            cpu_state_dict = model.state_dict()
            return cpu_state_dict, float(train_loss_history[-1])
        else:
            return None, None
    finally:
        # Cleanup - safely delete only if they exist
        if model is not None:
            # Move model to CPU before deleting to avoid CUDA errors
            if next(model.parameters(), None) is not None:
                model.cpu()
            del model
        if optimizer is not None:
            del optimizer
        torch.cuda.empty_cache()

def worker(task_id, gpu_id, model_architecture, permutation_free_settings, train_loader, test_loader,
           num_epochs, learning_rate, loss_fn, success_loss, convergence_threshold):
    try:
        # Set device before any CUDA operations
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        
        # Log open file descriptors at start
        fds_start = len(os.listdir('/proc/self/fd'))
        print(f"Task {task_id} start: open fds: {fds_start}")
        
        # Run training
        result, final_loss = train_single_network(model_architecture, permutation_free_settings, train_loader, test_loader,
                                                  num_epochs, learning_rate, loss_fn, device,
                                                  success_loss, convergence_threshold, task_id)
        if result is not None:
            result = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
            final_loss = float(final_loss)
        
        # Log open file descriptors at end
        fds_end = len(os.listdir('/proc/self/fd'))
        print(f"Task {task_id} end: open fds: {fds_end}")
        if fds_end > fds_start:
            print(f"Warning: Task {task_id} increased open fds by {fds_end - fds_start}")
        
        return result, final_loss
    except Exception as e:
        print(f"Worker error (Task {task_id}, GPU {gpu_id}): {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except:
                pass

def check_and_save_networks(tasks_successful, last_save_count, collected_networks, 
                           intermittent_save_size, amount_to_produce, output_dir, output_file_name):
    """Helper function to check if networks should be saved and perform the save if needed.
    """
    if intermittent_save_size > 0 and output_dir is not None:
        if tasks_successful % intermittent_save_size == 0 or tasks_successful == amount_to_produce:
            networks_to_save = {}
            for i in range(last_save_count + 1, tasks_successful + 1):
                key = f'network_{i}'
                if key in collected_networks:
                    networks_to_save[key] = collected_networks[key]
            
            save_path = f"{output_dir}/{output_file_name}"
            # Save the newly collected networks
            save_networks(save_path, networks_to_save, append=True)
            print(f"Intermittent save: Saved networks {last_save_count+1} to {tasks_successful} to {save_path}")
            return tasks_successful
    return last_save_count

def parallel_train_networks(model_architecture, permutation_free_settings, train_loader, test_loader,
                            num_epochs, learning_rate, loss_fn,
                            success_loss, convergence_threshold,
                            amount_to_produce, max_workers=4, 
                            output_dir=None, output_file_name='working_networks.pt', intermittent_save_size=0):
    collected_networks = {}
    tasks_submitted = 0
    tasks_successful = 0
    last_save_count = 0
    
    while tasks_successful < amount_to_produce:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            gpu_cycle = cycle(range(max_workers))
            while tasks_successful < amount_to_produce:
                # Submit a new task only if the target isn't reached
                gpu_id = next(gpu_cycle)
                future = executor.submit(worker, tasks_submitted, gpu_id, model_architecture, permutation_free_settings,
                                     train_loader, test_loader, num_epochs,
                                     learning_rate, loss_fn, success_loss,
                                     convergence_threshold)
                futures[future] = tasks_submitted
                tasks_submitted += 1
                
                for fut in list(futures):
                    if fut.done():
                        task_id = futures.pop(fut)
                        try:
                            result, final_loss = fut.result()
                            if result is not None:
                                network_name = f'network_{tasks_successful+1}'
                                collected_networks[network_name] = result
                                tasks_successful += 1
                                print(f"Network {tasks_successful} succeeded (Task {task_id}).")
                                
                                last_save_count = check_and_save_networks(
                                    tasks_successful, last_save_count, collected_networks,
                                    intermittent_save_size, amount_to_produce, output_dir, output_file_name
                                )
                        except Exception as e:
                            print(f"Task {task_id} encountered an error: {e}")
                        
            
            # Wait for any remaining futures to complete
            for fut in as_completed(futures):
                task_id = futures[fut]
                try:
                    result, final_loss = fut.result(timeout=1800)  # 30 minutes timeout
                    if result is not None:
                        network_name = f'network_{tasks_successful+1}'
                        collected_networks[network_name] = result
                        tasks_successful += 1
                except Exception as e:
                    print(f"Task {task_id} failed with: {e}")
        
        # Force Python garbage collection between pool restarts
        gc.collect()
    
    return collected_networks

def save_networks(save_path, networks_to_save, append=False):
    """Save networks to file, with option to append to existing file"""
    if append and os.path.exists(save_path):
        try:
            # Use context manager to ensure file is closed
            existing_networks = {}
            with open(save_path, 'rb') as f:
                existing_networks = torch.load(f)
            
            # Merge with new networks
            for key, value in networks_to_save.items():
                existing_networks[key] = value
            
            # Save merged networks with context manager
            with open(save_path, 'wb') as f:
                torch.save(existing_networks, f)
        except Exception as e:
            print(f"Warning: Failed to append to existing file: {e}")
            print(f"Saving only new networks instead.")
            with open(save_path, 'wb') as f:
                torch.save(networks_to_save, f)
    else:
        # Save without appending, using context manager
        with open(save_path, 'wb') as f:
            torch.save(networks_to_save, f)

def load_data(data_path, batch_size=1024):
    csv_data = pd.read_csv(data_path)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Train neural networks in parallel with permutation-free architecture')
    parser.add_argument('--data', type=str, default='data/simpleReg.csv', help='Path to data CSV file')
    parser.add_argument('--success-loss', type=float, default=0.1, help='Threshold for successful training')
    parser.add_argument('--convergence-threshold', type=float, default=1e-1, help='Threshold for determining convergence')
    parser.add_argument('--amount', type=int, default=100, help='Number of successful networks to produce')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs per network')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--output-dir', type=str, default='WorkingNetworks', help='Directory to save networks')
    parser.add_argument('--output-file-name', type=str, default='working_networks.pt', help='Name of the file to save networks')
    parser.add_argument('--intermittent-save-size', type=int, default=100, 
                        help='Save networks after collecting this many successful ones (0 to disable)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    mp.set_start_method('spawn')
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using up to {args.gpus} GPUs")
    
    # Load data
    train_loader, test_loader = load_data(args.data, args.batch_size)
    print("Data loaded from", args.data)
    print(f"Using batch size of {args.batch_size}")
    
    # Setup training parameters
    loss_fn = nn.MSELoss()
    
    # Create model architecture with fixed permutation-free settings
    use_masked = True
    itype = "masked"  # Options: "masked", "sparse", "static"
    freeze = True
    
    model_architecture = create_permutation_free_net
    
    # Store experiment metadata
    permutation_free_settings = {
        'use_masked': use_masked,
        'itype': itype,
        'freeze': freeze
    }
    experiment_metadata = {
        'date': datetime.now().isoformat(),
        'parameters': {
            'success_loss': args.success_loss,
            'convergence_threshold': args.convergence_threshold,
            'amount_to_produce': args.amount,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'max_workers': args.gpus,
            'data_path': args.data,
            'permutation_free_settings': permutation_free_settings,
            'intermittent_save_size': args.intermittent_save_size
        }
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we have existing networks to continue from
    existing_networks = {}
    save_path = f"{args.output_dir}/{args.output_file_name}"
    if os.path.exists(save_path):
        try:
            existing_networks = torch.load(save_path)
            print(f"Loaded {len(existing_networks)} existing networks from {save_path}")
            if existing_networks:
                # Find the highest network index
                max_idx = 0
                for key in existing_networks.keys():
                    if key.startswith('network_'):
                        try:
                            idx = int(key.split('_')[1])
                            max_idx = max(max_idx, idx)
                        except:
                            pass
                print(f"Will continue training from network_{max_idx+1}")
        except Exception as e:
            print(f"Failed to load existing networks: {e}")
            existing_networks = {}
    
    # Run training
    print(f"Starting training with parameters: {experiment_metadata['parameters']}")
    networks_state_dicts = parallel_train_networks(
        model_architecture,
        permutation_free_settings,
        train_loader,
        test_loader,
        args.epochs,
        args.lr,
        loss_fn,
        args.success_loss,
        args.convergence_threshold,
        args.amount,
        max_workers=args.gpus,
        output_dir=args.output_dir,
        output_file_name=args.output_file_name,
        intermittent_save_size=args.intermittent_save_size
    )
    
    # Merge with existing networks if we started with some
    if existing_networks:
        for key, value in existing_networks.items():
            if key not in networks_state_dicts:
                networks_state_dicts[key] = value
    
    # Save final results
    torch.save(networks_state_dicts, save_path)
    
    # Save metadata
    with open(f"{args.output_dir}/metadata.json", 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    print(f"Training complete. Saved {len(networks_state_dicts)} networks to {args.output_dir}")
    print(f"Metadata saved to {args.output_dir}/metadata.json")