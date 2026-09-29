from torch.utils.data import Dataset
import numpy as np
import torch
from snnpiece.analysis.causal_sets import get_causal_piece_from_recorded_data, turn_neuron_causal_sets_to_IDs, get_layer_indices, get_number_of_sets_per_neuron
from collections import Counter
from tqdm import tqdm
from snnpiece.datasets.YinYang import get_YinYang_dataloaders

### Count pieces on data samples
def scan_through_dataset(model, batchsize = 5000):
    """
    Count number of pieces the training data falls into for Yin-Yang dataset.
    """
    x_values = []
    y_values = []
    loader, _ = get_YinYang_dataloaders(batch_size=batchsize)

    results = []
    sizes = []
    model.cpu()
    model.eval()
    model.return_detailed_output = True
    print('Running network and extracting causal sets...')
    for inp,_ in tqdm(loader):
        _, data = model(inp)
        csets, csize = get_causal_piece_from_recorded_data(data)
        results += csets
        sizes += csize
        x_values += inp[:,0].detach().numpy().tolist() #CHANGE
        y_values += inp[:,1].detach().numpy().tolist()
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
        'alpha_values': x_values,
        'beta_values': y_values,
        'layer_IDs': layer_IDs,
        'lay_dict': lay_dicts,
        'neuron_dict': causal_set_IDs,
        'neuron_IDs': sets,
        'set_sizes': sizes,
        'num_sets': num_sets,
    }

    total_num_pieces = get_number_of_sets(output)
    piece_sizes = get_size_of_pieces(output)
    set_sizes = get_num_weights_in_sets(output)

    return output, total_num_pieces, piece_sizes, set_sizes

def get_number_of_sets(output):
    sets = set()
    for entry in output['layer_IDs']:
        sets.add(entry[-1])
    return len(sets)

def get_size_of_pieces(output):
    counts = np.array(list(Counter(np.array(output['layer_IDs'])[:,-1]).values()))
    counts = counts/np.sum(counts)*(np.pi*0.5**2)

    return counts

def get_num_weights_in_sets(output):
    num_layers = len(output['neuron_IDs'][0])
    neurons = [len(output['neuron_IDs'][0][i]) for i in range(num_layers)]

    IDs = [[set() for i in range(neurons[j])] for j in range(num_layers)]
    set_sizes = [[[] for i in range(neurons[j])] for j in range(num_layers)]

    for sample in range(len(output['neuron_IDs'])):
        for layer in range(len(output['neuron_IDs'][0])):
            for neuron in range(len(output['neuron_IDs'][sample][layer])):
                if output['neuron_IDs'][sample][layer][neuron] not in IDs[layer][neuron]:
                    set_sizes[layer][neuron].append(output['set_sizes'][sample][layer][neuron])    
                    IDs[layer][neuron].add(output['neuron_IDs'][sample][layer][neuron])

    return set_sizes

# Count pieces by randomly sampling the input space

class CustomRandomDataset(Dataset):
    """
    Dataset with random data points.
    """
    def __init__(self, dim, bound, num_samples, seed):
        np.random.seed(seed)
        print('Create dataset...')
        self.inputs = torch.Tensor((np.random.random(size=(num_samples, dim))-0.5)*2*bound)        

    def __getitem__(self, index):
        inp = self.inputs[index]
        
        return inp

    def __len__(self):
        return self.inputs.size(0)

def create_batched_random_data(dim, bound, num_samples, seed, batchsize):
    dataset = CustomRandomDataset(dim, bound, num_samples, seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)
    return dataloader
    
def scan_randomly(model, dim, bound, num_samples, seed, batchsize = 1000):
    """
    Count into how many different pieces random samples fall into.
    """
    alpha_values = []
    beta_values = []
    loader = create_batched_random_data(dim, bound, num_samples, seed, batchsize)

    results = []
    sizes = []
    print('Running network and extracting causal sets...')
    for inp in tqdm(loader):
        _, data = model(inp)
        csets, csize = get_causal_piece_from_recorded_data(data)
        results += csets
        sizes += csize
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