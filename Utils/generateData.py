import numpy as np
import pandas as pd
import argparse

def parse_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate regression data')
    parser.add_argument('--num_points', type=int, default=100000, 
                        help='Number of data points to generate (default: 100000)')
    parser.add_argument('--output_file', type=str, default='./data/simpleReg.csv',
                        help='Path to output CSV file (default: ./data/simpleReg.csv)')
    parser.add_argument('--dataset_type', type=str, default='reference', choices=['reference', 'phenotype'],
                        help='Type of dataset to generate: reference or phenotype (default: reference)')
    return parser.parse_args()

def generate_input_data(num_points):
    np.random.seed(12)
    a = np.random.randn(num_points, 1)
    b = np.random.randn(num_points, 1)
    return a, b

def target_function_reference(a, b):
    return (1/5) * a**2 - (1/10) * b**3

def target_function_phenotype(a, b):
    return ((1/5) * a**2 - (1/10) * b**3) + 3

def create_dataset(a, b, y):
    return pd.DataFrame({
        'a': a.flatten(),
        'b': b.flatten(),
        'y': y.flatten(),
    })

def save_dataset(dataset, output_file):
    dataset.to_csv(output_file, index=False)
    
def main():
    args = parse_arguments()
    
    # Generate input data
    a, b = generate_input_data(args.num_points)
    print(a.min(), a.max())
    
    # Generate dataset based on specified type
    if args.dataset_type == 'reference':
        y = target_function_reference(a, b)
    else:  # phenotype
        y = target_function_phenotype(a, b)
    
    dataset = create_dataset(a, b, y)
    save_dataset(dataset, args.output_file)

if __name__ == "__main__":
    main()
