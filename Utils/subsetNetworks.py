#!/usr/bin/env python3
import sys
import torch
import os

def main():
    # Ensure we have the correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: {} input_file.pt subset_size output_file.pt".format(sys.argv[0]))
        sys.exit(1)
    
    input_file = sys.argv[1]
    subset_size = int(sys.argv[2])
    output_file = sys.argv[3]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    # Load the networks from the input file
    print(f"Loading networks from '{input_file}'...")
    try:
        data = torch.load(input_file)
    except Exception as e:
        print(f"Error loading '{input_file}': {e}")
        sys.exit(1)
    
    # Check if the data is in the expected format
    if not isinstance(data, dict):
        print(f"Error: '{input_file}' does not contain a dictionary of networks.")
        sys.exit(1)
    
    # Find all network keys and sort them
    network_keys = sorted([k for k in data.keys() if k.startswith("network_")], 
                         key=lambda x: int(x.split('_')[1]))
    
    total_networks = len(network_keys)
    print(f"Found {total_networks} networks in '{input_file}'.")
    
    # Check if subset size is valid
    if subset_size <= 0:
        print("Error: Subset size must be greater than 0.")
        sys.exit(1)
    
    if subset_size > total_networks:
        print(f"Warning: Requested subset size ({subset_size}) is larger than the number of available networks ({total_networks}).")
        print(f"Using all {total_networks} networks.")
        subset_size = total_networks
    
    # Create a new dictionary with the first n networks
    subset_data = {}
    for i in range(subset_size):
        key = network_keys[i]
        new_key = f"network_{i+1}"  # Renumber networks from 1 to n
        subset_data[new_key] = data[key]
    
    # Save the subset to the output file
    try:
        torch.save(subset_data, output_file)
        print(f"Saved subset of {subset_size} networks to '{output_file}'.")
        
        # Verification check
        verification_data = torch.load(output_file)
        verify_count = sum(1 for key in verification_data.keys() if key.startswith("network_"))
        print(f"Verification: '{output_file}' contains {verify_count} networks.")
    except Exception as e:
        print(f"Error saving to '{output_file}': {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
