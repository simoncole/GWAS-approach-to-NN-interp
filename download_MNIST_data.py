"""
This script downloads the MNIST dataset to a specified directory.
Run this on your CPU node before training on your GPU node.
"""

import os
from torchvision import datasets, transforms

def download_mnist(data_dir="./MNIST_data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Download training and test sets (download=True)
    datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    print(f"MNIST dataset downloaded to {data_dir}")

if __name__ == "__main__":
    download_mnist()