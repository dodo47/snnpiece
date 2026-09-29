import numpy as np
from scipy.special import binom
import random
from tqdm import tqdm

def samplesFromDistribution(name, parameters, num_samples, dim, seed):
    '''
    Sample weight vectors from a given distribution.

    Parameters
    ----------
    name : str
        Distribution name, e.g. 'normal', 'uniform', 'cauchy'.
    parameters : sequence
        Parameters passed to the selected distribution.
    num_samples : int
        Number of weight vector samples to draw.
    dim : int
        Number of weights.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (int(num_samples * dim), dim) containing sampled weights.
    '''
    np.random.seed(seed)
    if name == 'shifted_uniform':
        return (np.random.uniform(size=(int(num_samples*dim), dim))-0.5)*2*parameters[1]+parameters[0]
    elif name == 'cauchy':
        return np.random.standard_cauchy(size=(int(num_samples*dim), dim))*parameters[1]+parameters[0]
    elif name == 'uniform':
        return np.random.uniform(0, parameters[1], size=(int(num_samples*dim), dim))+parameters[0]
    elif name == 'normal':
        return np.random.normal(parameters[0], parameters[1], size=(int(num_samples*dim), dim))
    elif name == 'lognormal':
        return np.random.lognormal(parameters[0], parameters[1], size=(int(num_samples*dim), dim))*parameters[2]
    elif name == 'pareto':
        return np.random.pareto(parameters[0], size=(int(num_samples*dim), dim))*parameters[1]
    elif name == 'binomial':
        return (np.random.binomial(1, parameters[0], (int(num_samples*dim), dim))-1/2)*2*parameters[1]
    elif name == 'xavier_normal':
        factor = np.sqrt(2/(dim+1))
        return samplesFromDistribution('normal', [0, factor], num_samples, dim, seed) 
    elif name == 'xavier_uniform':
        factor = np.sqrt(6/(dim+1))
        return samplesFromDistribution('shifted_uniform', [0, factor], num_samples, dim, seed)
    

def sampleWeightsForCausalPieces(name, parameters, num_samples, dim, seed, thresh=1):
    '''
    Estimate the number of causal pieces realised by a random weight vector.

    Parameters
    ----------
    name : str
        Weight distribution name.
    parameters : sequence
        Parameters for the distribution.
    num_samples : int
        Number of Monte Carlo samples per subset size.
    dim : int
        Number of variables.
    seed : int
        Random seed.
    thresh : float, optional
        Spike threshold.

    Returns
    -------
    tuple
        ``probs`` for each subset size and the piece estimate (given in relation to the max. number of pieces).
    '''
    dimMask = np.ones((int(num_samples*dim), dim))
    for i in range(1,dim):
        dimMask[(i-1)*int(num_samples):i*int(num_samples), i:] *= 0

    weight = samplesFromDistribution(name, parameters, num_samples, dim, seed)        
    strong_enough = np.sum(weight*dimMask, axis=1) >= thresh
    probs = np.mean(np.reshape(strong_enough, (dim, int(num_samples))), axis=1)

    k = np.arange(0, dim+1)
    binom_coeff = binom(dim, k)
    max_num_pieces_fraction = np.sum(probs*binom_coeff[1:])/(2**(dim)-1)

    return probs, max_num_pieces_fraction

def CausalPiecesFromSingleWeightMatrix(weights, num_samples, seed, thresh=1):
    '''
    Same as ``sampleWeightsForCausalPieces`` but evaluated for a single weight matrix.

    Parameters
    ----------
    weights : sequence
        Weight vector.
    num_samples : int
        Number of random subset draws per subset size.
    seed : int
        Random seed.
    thresh : float, optional
        Spike threshold.

    Returns
    -------
    tuple
        ``probs`` for each subset size and the piece estimate (given in relation to the max. number of pieces).
    '''
    random.seed(seed)
    dim = len(weights)

    probs = []
    for setsize in tqdm(range(1,dim+1)):
        probs.append(0)
        for _ in range(num_samples):
            subsample = random.sample(weights, setsize)
            isset = np.sum(subsample) >= thresh
            probs[-1] += isset
        probs[-1] /= num_samples
    probs = np.array(probs)

    k = np.arange(0, dim+1)
    binom_coeff = binom(dim, k)
    max_num_pieces_fraction = np.sum(probs*binom_coeff[1:])/(2**(dim)-1)

    return probs, max_num_pieces_fraction