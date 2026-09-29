import torch
import torch.nn as nn
from .regularisation import weight_constraint, normalize_gradients
from .custom_losses import ttfsLoss2021
from tqdm import tqdm

def train(model, device, train_loader, optimizer, loss,
          gamma = 0, normed_grads = False, alpha = 0.2, debug = False):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    device : torch.device
        The device on which the computation is done.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training data.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    loss : str
        The type of loss to be used. Either 'CE' (Cross-Entropy) or 'ttfs' (Time-to-First-Spike).
    gamma : float, optional
        The weight for the additional loss term for dead neurons, by default 0.0.
    normed_grads : bool, optional
        Whether to normalize the gradients, by default False.
    alpha : float, optional
        The scaling for the Time-to-First-Spike loss, by default 0.2.
    debug : bool, optional
        Print non-finite diagnostics and run backward with anomaly detection.

    Notes
    -----
    The freely chosen parameter alpha is taken from Goeltz et al. (2021).

    """
    model.train()
    if loss == 'CE':
        lossf = nn.CrossEntropyLoss()
    elif loss == 'ttfs':
        lossf = ttfsLoss2021(scaling=alpha*model.neuron_params['taus'])

    recording = {'before': [], 'after': []}
    for (data, target) in tqdm(train_loader, leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, data_before_grad_update = model(data)
        if debug and not torch.isfinite(output).all():
            print('ExpIF debug: nonfinite model output before loss')
            raise RuntimeError('nonfinite model output')
        loss = lossf(output, target)
        if debug and not torch.isfinite(loss):
            print('ExpIF debug: nonfinite loss')
            raise RuntimeError('nonfinite loss')
        if gamma != 0:
            loss += gamma*weight_constraint(model.IFweights, model.neuron_params['threshold'])
        if debug:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()
        if debug:
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print('ExpIF debug: nonfinite grad in', name)
                    raise RuntimeError(f'nonfinite grad in {name}')
        if normed_grads == True:
            normalize_gradients(model.IFweights, model.neuron_params['threshold'])
        optimizer.step()
        if debug:
            for name, p in model.named_parameters():
                if not torch.isfinite(p).all():
                    print('ExpIF debug: nonfinite parameter after step in', name)
                    raise RuntimeError(f'nonfinite parameter in {name}')
        if model.return_detailed_output == True:
            _, data_after_grad_update = model(data)
            recording['before'].append(data_before_grad_update)
            recording['after'].append(data_after_grad_update)
    
    return recording

def test(model, device, test_loader, loss):
    """
    Evaluate the model on a given test loader.

    Parameters
    ----------
    model : Module
        The model to be evaluated.
    device : torch.device
        The device on which the computations are performed.
    test_loader : DataLoader
        The DataLoader containing the test data.
    loss : str
        The type of loss that has been used. Either 'CE' (Cross-Entropy) or 'ttfs' (Time-to-First-Spike).
        Important, as the label neurons is being identified differently for different loss functions.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the test set.
    """
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, leave=False):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            if loss == 'ttfs':
                pred = output.argmin(dim=1)
            elif loss == 'CE':
                pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.numel()

    accuracy = 100.0 * correct / total

    return accuracy
