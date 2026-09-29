import torch

def normalize_gradients(params, thresh):
    '''
    Normalize weight gradients.
    
    During training, weight gradients can sometimes jump in their norm.
    Use this to rescale the gradients in cases where weights would drastically jump.
    '''
    for weight in params:
        gnorm = torch.norm(weight.grad)
        if gnorm > thresh:
            weight.grad.data /= gnorm
            weight.grad.data *= thresh
      
def weight_constraint(params, thresh):
    '''
    If the sum of weights to an output neuron does not exceed the threshold,
    this additional loss pushes the weights up to cross the threshold.
    
    Used to counter dead neurons.
    '''
    wloss = 0.
    for weight in params:
        weight_norm = (weight*(weight>0)).sum(0)
        wloss += ((thresh-weight_norm)*(weight_norm < thresh)).sum()
    return wloss