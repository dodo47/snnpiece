import numpy as np 
from tqdm import tqdm
import hashlib

#### General functions used for couting causal pieces

# Used for MNIST, FashionMNIST, EuroSAT
def collect_all_pieces(all_sets, causal_pieces): 
    """
    Collect all distinct causal pieces seen across a dataset.

    Parameters
    ----------
    all_sets : list
        Nested causal-set data indexed by sample, layer, and neuron.
    causal_pieces : set
        Set used to store unique causal-piece hashes.

    Returns
    -------
    None
        The provided ``causal_pieces`` set is updated in place.
    """
    num_samples = len(all_sets)
    num_layers = len(all_sets[0])

    for sample_id in tqdm(range(num_samples)):
        neurons = np.arange(0, len(all_sets[sample_id][-1]), 1)
        piece_name = ''
        for layer_id in range(num_layers-1, -1, -1):
            cpieces = all_sets[sample_id][layer_id]
            new_neurons = set()
            partial_name = []
            for neu in neurons:
                partial_name.append(cpieces[neu])
                new_neurons.update(set(cpieces[neu]))
            neurons = new_neurons 
            piece_name += ' ' + str(partial_name)
        hash_object = hashlib.md5(piece_name.encode())
        causal_pieces.add(hash_object.hexdigest()) 

# Used for Yin-Yang
def get_causal_piece_from_recorded_data(data):
    """
    Convert recorded causal-set metadata into per-sample, per-neuron causal sets.

    Parameters
    ----------
    data : dict
        Dictionary containing recorded causal-set information and spike times.

    Returns
    -------
    tuple
        Causal sets and set sizes for each sample and layer/neuron.
    """
    causal_sets = data['causal_sets']
    spike_times = data['spike_times']
    num_samples = len(causal_sets[0][0])
    set_per_neuron = [[] for i in range(num_samples)]
    set_sizes = [[] for i in range(num_samples)]
    for layer_id, cset in enumerate(causal_sets):
        for sample_id in range(num_samples):
            set_per_neuron[sample_id].append([])
            set_sizes[sample_id].append([])
            for neuron_id, max_index in enumerate(cset[0][sample_id]):
                if spike_times[layer_id][sample_id][neuron_id] > 4.5:
                    set_identifier = []
                else:
                    set_identifier = cset[1][sample_id][:max_index+1].tolist()
                    set_identifier.sort()
                set_per_neuron[sample_id][-1].append(set_identifier)
                set_sizes[sample_id][-1].append(len(set_identifier))
    return set_per_neuron, set_sizes

def turn_neuron_causal_sets_to_IDs(all_sets, neuron_causal_set_to_ID_dict): 
    """
    Turns a list of lists of lists of causal sets into a list of lists of IDs.
    
    Parameters
    ----------
    all_sets : list of lists of lists
        Causal sets per neuron per layer per sample.
    neuron_causal_set_to_ID_dict : dict
        Dictionary mapping causal sets to IDs.
    
    Returns
    -------
    IDs : list of lists of IDs
        IDs per neuron per layer per sample.
    """
    num_samples = len(all_sets)
    IDs = [[] for kk in range(num_samples)]
    for sample_id in tqdm(range(num_samples)):
        set_per_neuron = all_sets[sample_id]
        for layer_id in range(len(set_per_neuron)):
            layers = set_per_neuron[layer_id]
            IDs[sample_id].append([])
            for causal_set in layers:
                cset_name = str(causal_set)
                if layer_id > 0:
                    cset_name = str([np.array(IDs[sample_id][layer_id-1])[causal_set].tolist(), causal_set])
                if len(causal_set) == 0:
                    cset_name = str([])
                if cset_name not in neuron_causal_set_to_ID_dict.keys():
                    neuron_causal_set_to_ID_dict[cset_name] = len(neuron_causal_set_to_ID_dict)
                IDs[sample_id][-1].append(neuron_causal_set_to_ID_dict[cset_name])
    return IDs

def get_layer_indices(data, layer_indices_dict):
    """
    Calculates causal sets on the layer level.
    """
    transformed_data = []

    for i in tqdm(range(len(data))):
        transformed_data.append([])
        for j in range(len(data[i])):
            lay_state = str(data[i][j])
            if lay_state not in layer_indices_dict[j].keys():
                layer_indices_dict[j][lay_state] = len(layer_indices_dict[j])
            transformed_data[-1].append(layer_indices_dict[j][lay_state])
    return transformed_data

# Used for counting pieces on a hyperplane / line
def get_number_of_sets_per_neuron(neuron_IDs):
    """
    Count the number of distinct causal sets observed for each neuron.

    Parameters
    ----------
    neuron_IDs : list
        Nested list of causal-set IDs indexed by sample, layer, and neuron.

    Returns
    -------
    list
        Number of unique causal sets per layer and neuron.
    """
    set_counts = [[set() for i in range(len(neuron_IDs[0][j]))] for j in range(len(neuron_IDs[0]))]

    for samples in tqdm(neuron_IDs):
        for layers in range(len(samples)):
            for neurons in range(len(samples[layers])):
                set_counts[layers][neurons].add(samples[layers][neurons])
    for layers in range(len(set_counts)):
        for neurons in range(len(set_counts[layers])):
            set_counts[layers][neurons] = len(set_counts[layers][neurons]) 

    return set_counts
