#!/usr/bin/env python3
import sys
import torch
import os
import datetime

def main():
    # Ensure there is at least one input file and an output file.
    if len(sys.argv) < 3:
        print("Usage: {} input_file1.pt input_file2.pt ... output_file.pt".format(sys.argv[0]))
        sys.exit(1)
    
    # All arguments except the last are input files; the last is the output file.
    *input_files, output_file = sys.argv[1:]
    
    # Get current date and time for file naming with more readable format
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # Check if output file exists and load existing networks
    if os.path.exists(output_file):
        print(f"Output file '{output_file}' exists. Loading existing networks...")
        combined = torch.load(output_file)
        
        # Count existing networks
        existing_count = sum(1 for key in combined.keys() if key.startswith("network_"))
        print(f"Found {existing_count} existing networks.")
        
        # Find the highest network number to continue from
        highest_num = 0
        for key in combined.keys():
            if key.startswith("network_"):
                try:
                    num = int(key.split('_')[1])
                    highest_num = max(highest_num, num)
                except ValueError:
                    pass
        
        counter = highest_num
    else:
        combined = {}
        counter = 0
        print(f"Creating new output file '{output_file}'.")

    added_counter = 0
    renamed_files = []
    
    for file in input_files:
        # First load the data from the original file
        data = torch.load(file)
        
        # Create the new timestamped filename
        file_base, file_ext = os.path.splitext(file)
        timestamped_filename = f"{file_base}_{timestamp}{file_ext}"
        
        # Rename the actual file
        try:
            os.rename(file, timestamped_filename)
            print(f"Renamed '{file}' to '{timestamped_filename}'")
            renamed_files.append((file, timestamped_filename))
        except Exception as e:
            print(f"Error renaming '{file}': {e}")
            continue
        
        if isinstance(data, dict):
            # Iterate over all items in the dictionary.
            for key, value in data.items():
                # Create keys like 'network_1', 'network_2', etc.
                counter += 1
                new_key = f"network_{counter}"
                combined[new_key] = value
                added_counter += 1
        else:
            print(f"Warning: {timestamped_filename} does not contain a dictionary; skipping.")

    torch.save(combined, output_file)
    print(f"Added {added_counter} networks to '{output_file}'.")
    
    # Print summary of renamed files
    if renamed_files:
        print("\nSummary of renamed files:")
        for original, renamed in renamed_files:
            print(f"  {original} → {renamed}")
    
    # Verification check: Load the output file and count networks
    verification_data = torch.load(output_file)
    network_count = sum(1 for key in verification_data.keys() if key.startswith("network_"))
    print(f"Verification: '{output_file}' now contains {network_count} networks.")

if __name__ == '__main__':
    main()
