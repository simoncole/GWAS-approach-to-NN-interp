#!/usr/bin/env python
"""
This script uses multiprocessing to concurrently train neural networks with various architectures. 
Each worker is assigned to one GPU to train a network. Only networks that meet the success criteria are saved.
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
import json
from datetime import datetime
import gc
from concurrent.futures import as_completed
import concurrent.futures



class NeuralNetworkTrainer:
    """
    A class to handle parallel training of neural networks across multiple GPUs.
    Only networks that meet the success criteria are saved.
    
    This class is architecture-agnostic and can be used with any PyTorch model class.
    Provide the model class and any arguments required for instantiation.
    """
    
    def __init__(self, 
                 data_path,
                 model_architecture,
                 model_args=None,        
                 success_loss=0.1, 
                 convergence_threshold=1e-1, 
                 amount_to_produce=500,
                 epochs=20, 
                 learning_rate=0.1, 
                 max_gpus=4, 
                 output_dir='WorkingNetworks',
                 output_file_name='working_networks.pt',
                 intermittent_save_size=100,
                 batch_size=1024):
        """Initialize the trainer with configuration parameters."""
        self.data_path = data_path
        self.success_loss = success_loss
        self.convergence_threshold = convergence_threshold
        self.amount_to_produce = amount_to_produce
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_gpus = max_gpus
        self.output_dir = output_dir
        self.output_file_name = output_file_name
        self.intermittent_save_size = intermittent_save_size
        self.batch_size = batch_size
        
        # Set model architecture and arguments
        self.model_architecture = model_architecture
        self.model_args = model_args or {}
        
        # Runtime variables
        self.train_loader = None
        self.test_loader = None
        self.loss_fn = nn.MSELoss()
        self.collected_networks = {}
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot continue.")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Using up to {self.max_gpus} GPUs")
        print(f"Model architecture: {self.model_architecture.__name__}")

    def load_data(self):
        """Load and prepare data for training."""
        csv_data = pd.read_csv(self.data_path)
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

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        print("Data loaded from", self.data_path)
        print(f"Using batch size of {self.batch_size}")

    @staticmethod
    def train_single_network(model_class, model_args, train_loader, test_loader, num_epochs,
                           learning_rate, loss_fn, device, success_loss, convergence_threshold, task_id):
        """Train a single network and return its state dict if successful."""
        model = None
        optimizer = None
        try:
            # Create model instance based on whether it needs arguments
            if model_args:
                model = model_class(**model_args).to(device)
            else:
                model = model_class().to(device)
                
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
            elif not converged:
                # Returning None for loss history and None for the state dict means it did not converge
                return None, None
            else:
                # Returning None for the state dict and the loss history means it did not converge
                return None, float(train_loss_history[-1])
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

    @staticmethod
    def worker(task_id, gpu_id, model_class, model_args, train_loader, test_loader,
             num_epochs, learning_rate, loss_fn, success_loss, convergence_threshold):
        """Worker function to train a network on a specific GPU."""
        try:
            # Set device before any CUDA operations
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)  # Explicitly set device to avoid auto assignment issues
            device = torch.device("cuda:0")
            
            # Add more aggressive garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            result, final_loss = NeuralNetworkTrainer.train_single_network(
                model_class, model_args, train_loader, test_loader,
                num_epochs, learning_rate, loss_fn, device,
                success_loss, convergence_threshold, task_id
            )
            if result is not None and final_loss is not None:
                # Ensure all tensors in state_dict are on CPU
                result = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
                # Convert loss to Python native float
                final_loss = float(final_loss)

            return result, final_loss
        except Exception as e:
            print(f"Worker error (Task {task_id}, GPU {gpu_id}): {e}")
            import traceback
            traceback.print_exc()
            return None, None
        finally:
            # More robust cleanup
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    pass

    def check_and_save_networks(self, tasks_successful, last_save_count, collected_networks):
        """Helper function to check if networks should be saved and perform the save if needed."""
        if self.intermittent_save_size > 0 and self.output_dir is not None:
            if (tasks_successful % self.intermittent_save_size == 0 or 
                tasks_successful == self.amount_to_produce):
                networks_to_save = {}
                for i in range(last_save_count + 1, tasks_successful + 1):
                    key = f'network_{i}'
                    if key in collected_networks:
                        networks_to_save[key] = collected_networks[key]
                
                save_path = f"{self.output_dir}/{self.output_file_name}"
                # Save the newly collected networks
                self.save_networks(save_path, networks_to_save, append=True)
                print(f"Intermittent save: Saved networks {last_save_count+1} to {tasks_successful} to {save_path}")
                return tasks_successful
        return last_save_count

    def parallel_train_networks(self):
        """Train networks in parallel across multiple GPUs until target amount is reached."""
        collected_networks = {}
        tasks_submitted = 0
        tasks_successful = 0
        last_save_count = 0
        failed_networks = []
        with ProcessPoolExecutor(max_workers=self.max_gpus) as executor:
            futures = {}
            gpu_cycle = cycle(range(self.max_gpus))
            
            # Initially submit only max_gpus tasks
            initial_tasks = min(self.max_gpus, self.amount_to_produce)
            for _ in range(initial_tasks):
                gpu_id = next(gpu_cycle)
                future = executor.submit(
                    self.worker, 
                    tasks_submitted, 
                    gpu_id, 
                    self.model_architecture,
                    self.model_args,        
                    self.train_loader, 
                    self.test_loader, 
                    self.epochs,
                    self.learning_rate, 
                    self.loss_fn, 
                    self.success_loss,
                    self.convergence_threshold
                )
                futures[future] = tasks_submitted
                tasks_submitted += 1
            
            # Process results and submit new tasks
            while futures and tasks_successful < self.amount_to_produce:
                # Wait for the next task to complete
                done, _ = concurrent.futures.wait(
                    futures, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for fut in done:
                    task_id = futures.pop(fut)
                    try:
                        result, final_loss = fut.result()
                        if result is not None:
                            network_name = f'network_{tasks_successful+1}'
                            collected_networks[network_name] = result
                            tasks_successful += 1
                            print(f"Network {tasks_successful} succeeded (Task {task_id}). Progress: {tasks_successful}/{self.amount_to_produce}")
                            
                            last_save_count = self.check_and_save_networks(
                                tasks_successful, last_save_count, collected_networks
                            )
                        elif result is None and final_loss is not None:
                            # Failed due to loss being too high
                            assert final_loss > self.success_loss
                            failed_networks.append(final_loss)
                        else:
                            # Failed due to not converging
                            failed_networks.append(None)
                            
                    except Exception as e:
                        print(f"Task {task_id} encountered an error: {e}")
                    
                    # Submit a new task only if more networks are needed
                    if tasks_successful < self.amount_to_produce:
                        gpu_id = next(gpu_cycle)
                        future = executor.submit(
                            self.worker, 
                            tasks_submitted, 
                            gpu_id, 
                            self.model_architecture,
                            self.model_args,        
                            self.train_loader, 
                            self.test_loader, 
                            self.epochs,
                            self.learning_rate, 
                            self.loss_fn, 
                            self.success_loss,
                            self.convergence_threshold
                        )
                        futures[future] = tasks_submitted
                        tasks_submitted += 1
        
        # Force Python garbage collection after training completes
        gc.collect()
        
        return collected_networks, failed_networks

    @staticmethod
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

    def load_existing_networks(self):
        """Load existing networks from the save path if they exist."""
        existing_networks = {}
        save_path = f"{self.output_dir}/{self.output_file_name}"
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
        return existing_networks

    def create_experiment_metadata(self):
        """Create metadata for the current experiment."""
        metadata = {
            'date': datetime.now().isoformat(),
            'parameters': {
                'model_architecture': self.model_architecture.__name__,
                'success_loss': self.success_loss,
                'convergence_threshold': self.convergence_threshold,
                'amount_to_produce': self.amount_to_produce,
                'num_epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_gpus': self.max_gpus,
                'data_path': self.data_path,
                'intermittent_save_size': self.intermittent_save_size
            }
        }
        
        # Add model arguments if they exist
        if self.model_args:
            # Convert any tensor values to their native Python types for JSON serialization
            serializable_args = {}
            for key, value in self.model_args.items():
                if isinstance(value, torch.Tensor):
                    serializable_args[key] = value.tolist() if value.numel() > 1 else float(value)
                else:
                    serializable_args[key] = value
            metadata['parameters']['model_args'] = serializable_args
            
        return metadata

    def run(self):
        """Main execution method to run the training process."""
        # Initialize multiprocessing
        mp.set_start_method('spawn', force=True)
        
        # Record start time
        start_time = datetime.now()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Check for existing networks
        existing_networks = self.load_existing_networks()
        
        # Create experiment metadata
        experiment_metadata = self.create_experiment_metadata()
        experiment_metadata['start_time'] = start_time.isoformat()
        print(f"Starting training with parameters: {experiment_metadata['parameters']}")
        
        # Run the training
        networks_state_dicts, failed_networks = self.parallel_train_networks()
        
        # Record failed networks information
        num_failed_networks = len(failed_networks)
        
        # Calculate average loss for failed networks if any exist
        non_none_losses = [loss for loss in failed_networks if loss is not None]
        avg_failed_loss = 0
        if non_none_losses:
            avg_failed_loss = sum(non_none_losses) / len(non_none_losses)
        
        # Add failed networks information to metadata
        experiment_metadata['failed_networks_count'] = num_failed_networks
        experiment_metadata['failed_networks_non_convergance_count'] = len(failed_networks) - len(non_none_losses)
        experiment_metadata['failed_networks_average_loss'] = avg_failed_loss
        
        # Print failed networks information
        print(f"Number of failed networks: {num_failed_networks}")
        print(f"Number of failed networks with loss (not None): {len(non_none_losses)}")
        print(f"Average loss of failed networks (with loss): {avg_failed_loss:.6f}")
        
        # Record completion time and calculate elapsed time
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        # Add time information to metadata
        experiment_metadata['completion_time'] = end_time.isoformat()
        experiment_metadata['elapsed_time_seconds'] = elapsed_time.total_seconds()
        experiment_metadata['elapsed_time_formatted'] = str(elapsed_time)
        
        # Merge with existing networks if we started with some
        if existing_networks:
            for key, value in existing_networks.items():
                if key not in networks_state_dicts:
                    networks_state_dicts[key] = value
        
        # Save final results
        save_path = f"{self.output_dir}/{self.output_file_name}"
        torch.save(networks_state_dicts, save_path)
        
        # Save metadata
        with open(f"{self.output_dir}/metadata.json", 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        print(f"Training complete. Saved {len(networks_state_dicts)} networks to {self.output_dir}")
        print(f"Metadata saved to {self.output_dir}/metadata.json")
        print(f"Total training time: {elapsed_time}")
        
        return networks_state_dicts

