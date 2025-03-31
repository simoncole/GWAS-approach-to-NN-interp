#!/usr/bin/env python
"""
Driver script for NeuralNetworkTrainer class.
Provides a flexible command-line interface to train neural networks using different architectures.
"""

import os
import sys
import json
import importlib
import argparse
from datetime import datetime

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

# Import the NeuralNetworkTrainer class
from Classes.neural_network_trainer import NeuralNetworkTrainer


def parse_args():
    """Parse command-line arguments for neural network training."""
    parser = argparse.ArgumentParser(
        description='Train neural networks in parallel with flexible architecture support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset parameters
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--data', type=str, required=True,
                         help='Path to data CSV file')
    data_group.add_argument('--batch-size', type=int, default=1024,
                         help='Batch size for training')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--epochs', type=int, default=20,
                          help='Number of training epochs per network')
    train_group.add_argument('--lr', type=float, default=0.1,
                          help='Learning rate')
    train_group.add_argument('--success-loss', type=float, default=0.1,
                          help='Threshold for successful training')
    train_group.add_argument('--convergence-threshold', type=float, default=1e-1,
                          help='Threshold for determining convergence')
    train_group.add_argument('--amount', type=int, default=100,
                          help='Number of successful networks to produce')
    
    # Hardware parameters
    hw_group = parser.add_argument_group('Hardware Options')
    hw_group.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    
    # Output parameters
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-dir', type=str, default='WorkingNetworks',
                           help='Directory to save networks')
    output_group.add_argument('--output-file-name', type=str, default='working_networks.pt',
                           help='Name of the file to save networks')
    output_group.add_argument('--intermittent-save-size', type=int, default=10,
                           help='Save networks after collecting this many successful ones (0 to disable)')
    
    # Architecture parameters
    arch_group = parser.add_argument_group('Architecture Options')
    arch_group.add_argument('--architecture-module', type=str, required=True,
                         help='Python module containing the architecture class (e.g., "Architectures.permutation_free_architecture")')
    arch_group.add_argument('--architecture-class', type=str, required=True,
                         help='Name of the architecture class within the module (e.g., "PermutationFreeNet")')
    arch_group.add_argument('--architecture-args', type=str, default='{}',
                         help='JSON string of arguments to pass to the architecture (e.g., \'{"use_masked": true, "itype": "masked", "freeze": true}\')')
    
    args = parser.parse_args()
    
    # Parse architecture arguments from JSON
    try:
        args.architecture_args = json.loads(args.architecture_args)
        # Convert string "true"/"false" to Python bool if needed
        for key, value in args.architecture_args.items():
            if isinstance(value, str):
                if value.lower() == "true":
                    args.architecture_args[key] = True
                elif value.lower() == "false":
                    args.architecture_args[key] = False
    except json.JSONDecodeError:
        print(f"Error: Could not parse architecture arguments: {args.architecture_args}")
        print("Please provide a valid JSON string.")
        sys.exit(1)
    
    return args


def load_architecture(module_name, class_name):
    """
    Dynamically import an architecture class from a specified module.
    
    Args:
        module_name: String name of the module (e.g., "Architectures.permutation_free_architecture")
        class_name: String name of the class (e.g., "PermutationFreeNet")
        
    Returns:
        The architecture class
    """
    try:
        module = importlib.import_module(module_name)
        arch_class = getattr(module, class_name)
        return arch_class
    except (ImportError, AttributeError) as e:
        print(f"Error loading architecture: {e}")
        print(f"Could not import {class_name} from {module_name}")
        sys.exit(1)


def main():
    """Main function to run neural network training."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load architecture class
    print(f"Loading architecture class {args.architecture_class} from {args.architecture_module}")
    architecture_class = load_architecture(args.architecture_module, args.architecture_class)
    
    # Print run configuration
    print("\n====== Neural Network Training Configuration ======")
    print(f"Starting at: {datetime.now().isoformat()}")
    print(f"Architecture: {args.architecture_class} from {args.architecture_module}")
    print(f"Architecture arguments: {args.architecture_args}")
    print(f"Training {args.amount} networks with {args.gpus} GPUs")
    print(f"Learning rate: {args.lr}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Success threshold: {args.success_loss}, Convergence threshold: {args.convergence_threshold}")
    print(f"Output directory: {args.output_dir}/{args.output_file_name}")
    print("=" * 50 + "\n")
    
    # Create trainer
    trainer = NeuralNetworkTrainer(
        data_path=args.data,
        model_architecture=architecture_class,
        model_args=args.architecture_args,
        success_loss=args.success_loss,
        convergence_threshold=args.convergence_threshold,
        amount_to_produce=args.amount,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_gpus=args.gpus,
        output_dir=args.output_dir,
        output_file_name=args.output_file_name,
        intermittent_save_size=args.intermittent_save_size,
        batch_size=args.batch_size
    )
    
    # Run training
    print("Starting training...")
    networks = trainer.run()
    
    print(f"\nTraining completed with {len(networks)} successful networks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
