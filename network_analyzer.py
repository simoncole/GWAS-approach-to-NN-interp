from typing import Union
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class NetworkAnalyzer:

    def __init__(self, model_architecture: nn.Module, amount_to_produce: int, success_loss: float, convergence_threshold: float = None, max_attempts: float = None):
        """
        Initializes the NetworkAnalyzer class with the required model architecture and configuration.

        Args:
            model_architecture (nn.Module): A PyTorch neural network model class specifying the architecture.
            amount_to_produce (int): Number of good neural networks to generate based on success criteria.
            success_loss (float): Loss amount that qualifies a good network. 
            convergence_threshold (float, optional): Maximum allowable difference in loss between the last two epochs 
                                            of training to determine if the neural network has converged.
            max_attempts (float, optional): Maximum number of attempts to produce a successful network, defaults
                                            to 130% of `amount_to_produce` if not specified.
        """
        self.model_architecture = model_architecture
        self.amount_to_produce = amount_to_produce
        self.success_loss = success_loss if success_loss is not None else self.__default_success_criteria__()
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else 1000000 #Making it a huge number so it always works
        self.max_attempts = max_attempts if max_attempts is not None else self.__default_max_attempts__()
        
        # Intialize these values to keep track of networks attempts
        self.attempt = 0  
        self.success_count = 0  

        # Array to keep track of loss
        self.train_loss_history = []
        self.test_loss_history = []

        # Ensure directories for saving networks exist
        self.working_dir = "WorkingNetworks"
        self.broken_dir = "BrokenNetworks"
        self.working_networks = {}
        self.broken_networks = {}

        # Set up directories
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)  # Create directory if it doesn't exist
        if not os.path.exists(self.broken_dir):
            os.makedirs(self.broken_dir)

    def __default_success_criteria__(self):
        """
        Provides a default success criteria if none is specified by the user.
        The default is a loss value of 1.0.

        Returns:
            float: The default success criteria (loss value).
        """
        return 1.0

    def __default_max_attempts__(self):
        """
        Sets the maximum number of attempts to produce successful networks to 130% of the amount requested,
        if not specified by the user.

        Returns:
            int: The calculated maximum number of attempts.
        """
        return int(self.amount_to_produce * 1.3)

    def __show_loss__(self, epoch: int, loss_value: float):
        """
        Prints the loss value for a specific epoch during training.

        Args:
            epoch (int): The current epoch number during training.
            loss_value (float): The loss value for the current epoch.
        """
        print(f"Epoch [{epoch+1}], Loss: {loss_value:.4f}")

    def __test_network__(self, model: nn.Module, test_loader: DataLoader, loss_fn: torch.nn.Module, device):
        """
        Tests the network using the testing dataset, and prints the Test loss

        Args:
            model (nn.Module): The trained PyTorch model to evaluate.
            test_loader (DataLoader): DataLoader for the test dataset.
            loss_fn (nn.Module): The loss function used to compute the loss.
            device (torch.device): The device on which to perform the evaluation.
        """
        model.eval()
        test_loss = 0.0

        print("Testing for the epoch")

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()

        avg_loss = test_loss / len(test_loader)
        self.test_loss_history.append(avg_loss)
        print(f"Test Loss: {avg_loss:.4f}")

    def __train_network__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, optimizer, loss_fn: torch.nn.Module, device, test: bool):
        """
        Trains a single network model over a specified number of epochs with early stopping.

        Args:
            model (nn.Module): The PyTorch neural network model to train.
            train_loader (DataLoader): The DataLoader for the training dataset.
            test_loader (DataLoader): The DataLoader for the test dataset.
            num_epochs (int): Number of epochs to train the model.
            optimizer (Optimizer): The optimizer used to update model weights.
            loss_fn (torch.nn.Module): The loss function used to compute the training loss.
            device: The GPU to be used.
        """
        
        best_loss = None
        epochs_no_improve = 0
        early_stopping_patience = None
        early_stopping_min_delta = 0.0001

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                input, target = batch
                input, target = input.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            self.__show_loss__(epoch, avg_loss)
            self.train_loss_history.append(avg_loss)

            if test:
                self.__test_network__(model, test_loader, loss_fn, device)
                
    def __train_network_parallel__(self, model: nn.Module, train_loader: DataLoader, num_epochs: int, optimizer, loss_fn: torch.nn.Module, device):
        """
        Trains a network using data parallelism across multiple GPUs.
        """
        # Wrap model in DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        model = model.to(device)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                input, target = batch
                # Move data to GPU(s)
                input, target = input.to(device), target.to(device)
                
                optimizer.zero_grad()
                # Forward pass will automatically distribute across GPUs when using DataParallel
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            self.__show_loss__(epoch, avg_loss)
            self.train_loss_history.append(avg_loss)

    def __check_success__(self, network: torch.nn.Module ):
        
        if self.train_loss_history[-1] <= self.success_loss and (self.train_loss_history[-1] - self.train_loss_history[-2] < self.convergence_threshold):
            self.working_networks[f'network_{self.success_count+1}'] = network.state_dict()
            self.success_count += 1
        else:
            self.broken_networks[f'network_{self.success_count+1}_{self.attempt}'] = network.state_dict()
            self.attempt += 1 

    def __get_device__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using {device} device!")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using {device} device!")
        else:
            device = torch.device("cpu")
            print("No GPU device available. Using CPU.")
        return device
    
    def __save_networks__(self):
        torch.save(self.working_networks, f'{self.working_dir}/working_networks.pt')
        torch.save(self.broken_networks, f'{self.broken_dir}/broken_networks.pt')

    def __visualize_loss__(self, train, test):
        if train or test:
            epochs = range(1, max(len(train), len(test)) + 1)
            if train:
                plt.plot(epochs, train, label="Training Loss", marker="o")
            if test:
                plt.plot(epochs, test, label="Testing Loss", marker="o")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.title("Training vs Testing Loss")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No loss history to plot.")

    def check_loss(self,  train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module, learning_rate):
        device = self.__get_device__()
        network_to_be = self.model_architecture()
        optimizer = torch.optim.SGD(network_to_be.parameters(), lr=learning_rate)
        network_to_be.to(device)
        self.__train_network__(network_to_be, train_loader, test_loader, num_epochs, optimizer, loss_fn, device, True)
        self.__visualize_loss__(self.train_loss_history, self.test_loss_history)
        self.train_loss_history = []
        self.test_loss_history = [] 
    
    def generate_networks(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module, learning_rate):
        device = self.__get_device__()
        while self.success_count < self.amount_to_produce:
            print(f"Training Network {self.success_count+1}")
            if self.attempt >= self.max_attempts:
                print("Error: Maximum number of attempts reached without meeting success criteria.")
                break
            
            network_to_be = self.model_architecture()
            optimizer = torch.optim.SGD(network_to_be.parameters(), lr=learning_rate)
            network_to_be.to(device)
            self.__train_network__(network_to_be, train_loader, test_loader, num_epochs, optimizer, loss_fn, device, False)
            self.__check_success__(network_to_be)
            
        if self.success_count == self.amount_to_produce:
            print(f"Successfully trained {self.success_count} networks.")
        else:
            print(f"{self.success_count} successful networks created out of {self.amount_to_produce}.")

        print(f"Attemps:{self.attempt}, successfull networks: {self.success_count}")
        self.__save_networks__()
        self.attempt = 0
        self.success_count = 0
    
    def generate_networks_parallel(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module, learning_rate):
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA GPU not available. Cannot train in Parallel")
        else:
            device = torch.device("cuda")
            print(f"Using {device} device!")

        while self.success_count < self.amount_to_produce:
            print(f"Training Network {self.attempt+1}")
            if self.attempt >= self.max_attempts:
                print("Error: Maximum number of attempts reached without meeting success criteria.")
                break

            network_to_be = self.model_architecture()
            optimizer = torch.optim.SGD(network_to_be.parameters(), lr=learning_rate)
            network_to_be = torch.nn.DataParallel(network_to_be)
            self.__train_network_parallel__(network_to_be, train_loader, num_epochs, optimizer, loss_fn, device)
            self.__check_success__(network_to_be)
            self.attempt += 1

        if self.success_count == self.amount_to_produce:
            print(f"Successfully trained {self.success_count} networks.")
        else:
            print(f"{self.success_count} successful networks created out of {self.amount_to_produce}.")

        print(f"Attemps:{self.attempt}, successfull networks: {self.success_count}")
        print("Resetting internal counter of networks...")
        self.attempt = 0
        self.success_count = 0