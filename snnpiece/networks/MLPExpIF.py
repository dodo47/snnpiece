import torch
import numpy as np
from ..layers import LinearExpIF, LinearDecoder

class MLPExpIF(torch.nn.Module):
    def __init__(self, layer_dim, seed, threshold = 1., taus = 0.1, last_spike = 5, 
                 positive_weights = False, max_delays = 0, return_detailed_output = False,
                 decoder = 'linear', trainable_decoder = False, init_type = 'xavier',
                 init_params = None, spike_epsilon = 1e-3, debug = False):
        super().__init__()
        # set seed for reproducibility
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # generate seed for each layer init
        layer_seeds = np.random.randint(1e6, size=len(layer_dim)-1)
        self.decoder = decoder
        self.debug = debug

        # network dimensions
        self.layer_dim = layer_dim

        # init params
        if init_params is not None:
            if not isinstance(init_params[0], list):
                init_params = [init_params]*len(layer_dim)

        # bundle neuron parameters that are the same for each layer 
        self.neuron_params = {'threshold': threshold, 'taus': taus, 'last_spike': last_spike,
                        'positive_weights': positive_weights, 'max_delays': max_delays,
                        'spike_epsilon': spike_epsilon, 'debug': debug}

        # first entry is the number of input neurons
        self.input_dim = layer_dim[0]

        # output format (just spikes, or spikes and causal set size)
        self.return_detailed_output = return_detailed_output

        ## Construct the network
        # add layers in a ModuleList
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_dim)-2):
            self.layers.append(LinearExpIF(layer_dim[i], layer_dim[i+1], seed = layer_seeds[i], **self.neuron_params, init_type=init_type, init_params=init_params[i]))
        # add last layer
        if decoder == 'linear':
            self.layers.append(LinearDecoder(layer_dim[-2], layer_dim[-1], trainable=trainable_decoder))
        elif decoder == 'ttfs':
            self.layers.append(LinearExpIF(layer_dim[-2], layer_dim[-1], seed = layer_seeds[-1], **self.neuron_params, init_type=init_type, init_params=init_params[-1]))
        else:
            raise Exception('Decoder {} not implemented'.format(decoder))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        
        recorder = {'spike_times': [], 'causal_sets': []}
        for layer_idx, layer in enumerate(self.layers):
            x, cset_data = layer(x)
            if self.debug and not torch.isfinite(x).all():
                print('MLPExpIF debug: nonfinite activation after layer', layer_idx)
                raise RuntimeError('nonfinite activation in MLPExpIF forward')
            if self.return_detailed_output == True:
                recorder['spike_times'].append(x.detach().cpu().numpy())
                if cset_data is not None:
                    recorder['causal_sets'].append(cset_data) 
                    
        return x, recorder

    @property
    def IFweights(self):
        IFweights = []

        offset = 0
        if self.decoder == 'linear':
            offset = -1

        for i in range(len(self.layers)+offset):
            extracted_weight = self.layers[i].w.weight
            if self.layers[i].positive_weights == True:
                extracted_weight = self.layers[i].constrain_weights(extracted_weight)
            IFweights.append(extracted_weight)

        return IFweights 
