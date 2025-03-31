import torch
from torch import nn
from .permutation_free import get_permfree_2to1_regression, create_masked_network

class PermutationFreeNet(nn.Module):
    def __init__(self, use_masked=False, itype="masked", freeze=True):
        super(PermutationFreeNet, self).__init__()
        if use_masked:
            self.network = create_masked_network(itype=itype, freeze=freeze)
        else:
            self.network = get_permfree_2to1_regression()
    
    def forward(self, x):
        return self.network(x)
    
    # Override the state_dict method to ensure CPU tensors
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # Create a new state dict with CPU tensors
        cpu_state = {}
        for key, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                cpu_state[key] = tensor.detach().cpu()
            else:
                cpu_state[key] = tensor
        return cpu_state
