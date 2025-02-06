"""
Minimizing the number of ways a neural network can be rearanged s.t. the probability
distributions are unique for each weight in the hidden layers. 
"""

# Describes which weights are allowed to change.
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

MASKS = [
    np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array([[1, 1, 1, 1, 1, 1]], dtype=bool),
]


def get_permfree_2to1_regression():
    "This network allows for permutation-free connectivity."
    return nn.Sequential(
        nn.Linear(2, 3),
        nn.ELU(),
        nn.Linear(3, 6),
        nn.ELU(),
        nn.Linear(6, 6),
        nn.ELU(),
        nn.Linear(6, 6),
        nn.ELU(),
        nn.Linear(6, 1),
    )


def get_init_weights_by_mask(mask: np.ndarray) -> torch.Tensor:
    """
    Initialize a sparse matrix of weights. Weights are sampled from a normal
    distribution. The std_dev = 1.0 / sqrt(n) where n is the
    number of active weights in the layer.
    """
    group_size = mask.T[0].sum()
    stdv = 1.0 / np.sqrt(group_size)
    weight_values = np.random.uniform(-stdv, stdv, (group_size, mask.shape[1]))
    weights = np.zeros_like(mask, dtype=np.float64)
    weights[mask] = weight_values.flatten()
    return torch.tensor(weights, dtype=torch.float32)


def get_freeze_weight_hook(mask):
    "Get a gradient hook that freezes the weights according to a mask"

    def grad_hook(grad):
        return grad * mask

    return grad_hook


def create_masked_network(
    itype: Literal["masked", "sparse", "static"] = "masked",
    masks: list[np.ndarray] = None,
    freeze=True,
):
    masks = masks or MASKS
    model: torch.Module = get_permfree_2to1_regression()
    with torch.no_grad():
        for j, mask in enumerate(masks):
            if freeze:
                model[j * 2].weight.register_hook(get_freeze_weight_hook(mask))
            if itype == "masked":
                w = model[j * 2].weight * mask
            elif itype == "sparse":
                w = get_init_weights_by_mask(mask)
            elif itype == "static":
                continue
            else:
                raise Exception(f"Invalid init type of {itype}")
            model[j * 2].weight.copy_(w)
    return model
