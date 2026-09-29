from snnpiece.analysis.causal_sets import collect_all_pieces, get_causal_piece_from_recorded_data
from snnpiece.datasets.EuroSAT import get_EuroSAT_dataloaders
from snnpiece.datasets.FashionMNIST import get_FashionMNIST_dataloaders
from snnpiece.datasets.MNIST import get_MNIST_dataloaders
from tqdm import tqdm

# Scan through the dataset: MNIST, FashionMNIST, EuroSAT
def scan_through_dataset(model, dataset_name, batchsize = 2000):
    """
    Count number of pieces for training data of MNIST, FashionMNIST, or EuroSAT.
    """
    if dataset_name == 'MNIST':
        loader, _ = get_MNIST_dataloaders('data', batch_size=batchsize)
    elif dataset_name == 'FashionMNIST':
        loader, _ = get_FashionMNIST_dataloaders('data', batch_size=batchsize)
    elif dataset_name == 'EuroSAT':
        loader, _ = get_EuroSAT_dataloaders('data', batch_size=batchsize)
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset_name}')
    model.cpu()
    model.eval()
    model.return_detailed_output = True
    print('Running network and extracting causal sets...')
    causal_pieces = set()
    for inp,_ in tqdm(loader):
        _, data = model(inp)
        csets, _ = get_causal_piece_from_recorded_data(data)
        collect_all_pieces(csets, causal_pieces)

    return len(causal_pieces)