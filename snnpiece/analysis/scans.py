import numpy as np 
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from .causal_sets import get_causal_piece_from_recorded_data, turn_neuron_causal_sets_to_IDs, get_layer_indices, get_number_of_sets_per_neuron

##############################
    # Routines for scanning through a plane or line in the input space
##############################

# Count pieces on a hyperplane

def scan_through_plane(model, origin, direction0, direction1, batchsize = 1000, steps = 1000, device='cuda'):
    """
    Evaluate causal pieces in a 2D plane of the input space.

    Parameters
    ----------
    model : nn.Module
        Neural network.
    origin, direction0, direction1 : array-like
        Basis vectors defining the scan region in input space.
    batchsize : int, optional
        Batch size used for evaluation.
    steps : int, optional
        Number of grid points along each direction.
    device : str, optional
        Device on which to run the model.

    Returns
    -------
    dict
        Dictionary containing sampled inputs, indexed causal sets, and set sizes.
    """
    alpha_values = []
    beta_values = []
    loader = create_batched_data(origin, direction0, direction1, steps, batchsize)

    results = []
    sizes = []
    print('Running network and extracting causal sets...')
    for inp, alpha, beta in tqdm(loader):
        _, data = model(inp)
        csets, csize = get_causal_piece_from_recorded_data(data)
        results += csets
        sizes += csize
        alpha_values.append(alpha.detach().numpy()) #CHANGE
        beta_values.append(beta.detach().numpy())
    print('Index causal sets...')
    causal_set_IDs = {'[]': -1}
    sets = turn_neuron_causal_sets_to_IDs(results, causal_set_IDs)

    print('Reducing to layerwise sets...')
    layers = model.layer_dim
    lay_dicts = [{}]*(len(layers)-1)
    layer_IDs = get_layer_indices(sets, lay_dicts)

    print('Count causal sets per neuron...')
    num_sets = get_number_of_sets_per_neuron(sets)

    output = {
        'alpha_values': alpha_values,
        'beta_values': beta_values,
        'layer_IDs': layer_IDs,
        'lay_dict': lay_dicts,
        'neuron_dict': causal_set_IDs,
        'neuron_IDs': sets,
        'set_sizes': sizes,
        'num_sets': num_sets,
    }
    return output

# Count pieces along a line

def scan_through_line(model, origin, direction0, steps = 1000):
    """
    Evaluate causal pieces along a 1D line in the input space. Also stores spike times.

    Parameters
    ----------
    model : nn.Module
        Network to evaluate.
    origin, direction0 : array-like
        Endpoint and direction defining the line segment.
    steps : int, optional
        Number of sample points along the line.

    Returns
    -------
    dict
        Output dictionary containing alpha values, causal-set IDs, and spike times.
    """
    alpha_values = np.linspace(0,1, steps)

    results = []
    sizes = []
    spikes = []
    print('Running network and extracting causal sets...')
    for alpha in tqdm(alpha_values):
        inp = origin + alpha*(direction0-origin)
        _, data = model(inp)
        csets, csize = get_causal_piece_from_recorded_data(data)
        results += csets
        sizes += csize
        spikes.append(data['spike_times'])
    print('Index causal sets...')
    causal_set_IDs = {'[]': -1}
    sets = turn_neuron_causal_sets_to_IDs(results, causal_set_IDs)

    print('Reducing to layerwise sets...')
    layers = model.layer_dim
    lay_dicts = [{}]*(len(layers)-1)
    layer_IDs = get_layer_indices(sets, lay_dicts)

    print('Count causal sets per neuron...')
    num_sets = get_number_of_sets_per_neuron(sets)

    output = {
        'alpha_values': alpha_values,
        'layer_IDs': layer_IDs,
        'lay_dict': lay_dicts,
        'neuron_dict': causal_set_IDs,
        'neuron_IDs': sets,
        'set_sizes': sizes,
        'spike_times': spikes,
        'num_sets': num_sets,
    }
    return output

### Helper functions for creating samples of the input space on a hyperplane

class CustomDataset(Dataset):
    """
    This dataset enumerates points of the form
    ``origin + alpha * (direction0 - origin) + beta * (direction1 - origin)``
    for a grid of alpha/beta values.
    """
    def __init__(self, origin, direction0, direction1, steps):
        alpha_values = []
        beta_values = []
        inputs = []
        print('Create dataset...')
        for alpha in tqdm(np.linspace(0,1,steps)):
            for beta in np.linspace(0,1,steps):
                alpha_values.append(alpha)
                beta_values.append(beta)
                inp = origin + alpha*(direction0-origin) + beta*(direction1-origin)
                inputs.append(inp)
        self.inputs = torch.vstack(inputs)
        self.alphas = torch.Tensor(alpha_values)
        self.betas = torch.Tensor(beta_values)

    def __getitem__(self, index):
        inp = self.inputs[index]
        alpha = self.alphas[index]
        beta = self.betas[index]

        return inp, alpha, beta

    def __len__(self):
        return self.inputs.size(0)

def create_batched_data(origin, direction0, direction1, steps, batchsize):
    """
    Create a dataloader for ``scan_through_plane``.

    Parameters
    ----------
    origin, direction0, direction1 : array-like
        Input-space points defining the scanning plane.
    steps : int
        Number of grid points per axis.
    batchsize : int
        Batch size for iteration.

    Returns
    -------
    DataLoader
        Data loader yielding (input, alpha, beta) tuples.
    """
    dataset = CustomDataset(origin, direction0, direction1, steps)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)
    return dataloader