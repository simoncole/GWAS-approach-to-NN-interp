import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def plot_functions(file1_path, file2_path, output_path):
    # Read the CSV files
    data1 = pd.read_csv(file1_path)
    data2 = pd.read_csv(file2_path)
    
    # Create a single 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot both functions on the same graph
    scatter1 = ax.scatter(data1['a'], data1['b'], data1['y'], 
                         c='blue', alpha=0.6, label='Function 1')
    scatter2 = ax.scatter(data2['a'], data2['b'], data2['y'], 
                         c='red', alpha=0.6, label='Function 2')
    
    # Set labels and title
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('y')
    ax.set_title('Comparison of Functions')
    
    # Add legend
    ax.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot 3D comparison of two functions')
    parser.add_argument('--file1', type=str, required=True,
                        help='Path to first CSV file')
    parser.add_argument('--file2', type=str, required=True,
                        help='Path to second CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the output plot')
    
    args = parser.parse_args()
    
    plot_functions(args.file1, args.file2, args.output)

if __name__ == "__main__":
    main()