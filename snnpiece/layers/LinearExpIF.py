import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

class LinearExpIF(nn.Module):
    def __init__(self, input_dim, output_dim, seed = None, threshold = 1., taus = 0.1,
                 last_spike = None, positive_weights = False, max_delays = 0,
                 init_type = 'xavier_normal', init_params = None, spike_epsilon = 1e-3,
                 debug = False):
        '''
        Analytical forward model of the IF model with exponential synapses, as used in Mostafa (2017),
        see https://arxiv.org/abs/1606.08165 (or also https://arxiv.org/abs/2104.13398).

        Version for fully connected layer.

        Args:
        input_dim: number of inputs (int)
        output_dim: number of neurons (int)
        seed: for reproducibility (int)
        threshold: spike threshold of output neurons (float)
        taus: synaptic time constant (float)
        last_spike: which spike time to assign when neurons do not fire at all (float)
        positive_weights: whether to constrain weights to be positive (bool)
        max_delays: maximum trainable input and output spike delays (float >= 0). 0: no delay.
        spike_epsilon: safety margin for the denominator sum_i w_i - threshold (float > 0).
        debug: print internal validity and dead-neuron stats during forward passes (bool)
        '''
        super().__init__()
        # set seed for reproducibility
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed+9999)
        # network dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # type of network initialization
        self.init_type = init_type
        self.init_params = init_params

        # neuron parameters
        self.threshold = torch.as_tensor(threshold)
        self.taus = torch.as_tensor(taus)
        # if a neuron does not spike, this value is used as its spike time
        self.last_spike = torch.as_tensor(last_spike)
        self.spike_epsilon = torch.as_tensor(spike_epsilon)
        self.debug = debug

        # whether to constrain weights to be positive   
        self.positive_weights = positive_weights
        # weight matrix
        self.w = torch.nn.Embedding(input_dim, output_dim)
        self.init_weights()

        # we have to build the causal set one input spike at a time, e.g.,
        # C = (t_0), C = (t_0, t_1), ...
        # we batch this by creating all possible causal sets at once (zeroing out components)
        # with a mask, e.g., for input_dim = 3: mask = ((1,0,0), (1,1,0), (1,1,1))
        mask = torch.tril(torch.ones(input_dim, input_dim))
        # make the mask transferable to cuda device
        self.register_buffer('mask', mask)

        # whether to add trainable delays to the model
        self.max_delays = max_delays
        if self.max_delays != 0:
            self.input_delay = torch.nn.Parameter(torch.zeros(self.input_dim))
            self.output_delay = torch.nn.Parameter(torch.zeros(self.output_dim))

    def init_weights(self):
        '''
        Initialize weights such that for each output neuron i,
        the sum of the incoming weights succeeds the threshold.
        '''
        nneurons, dim = self.w.weight.data.shape
        # constrain weights to be positive if desired
        if self.positive_weights == True:
            if self.init_type == 'uniform_layerwise':
                self.w.weight.data = torch.Tensor(np.random.uniform(0, self.init_params[0]*nneurons**(-self.init_params[1]), (nneurons, dim)) + self.init_params[2]*nneurons**(-self.init_params[3]))
            elif self.init_type == 'pareto_layerwise':
                 self.w.weight.data = torch.Tensor(np.random.pareto(self.init_params[0], size=(nneurons, dim))*self.init_params[1]*nneurons**(-self.init_params[2]))
            elif self.init_type == 'lognormal_layerwise':
                self.w.weight.data = torch.Tensor(np.random.lognormal(self.init_params[0]*nneurons**(-self.init_params[1]), 1, size=(nneurons, dim))*self.init_params[2]*nneurons**(-self.init_params[3]))
        else: 
            if self.init_type == 'normal':
                self.w.weight.data = torch.normal(self.init_params[0], self.init_params[1], size=(nneurons,dim))
            elif self.init_type == 'uniform_layerwise':
                self.w.weight.data = torch.Tensor((np.random.uniform(0, 1, (nneurons, dim)) - 0.5)*2*self.init_params[0]*nneurons**(-self.init_params[1]) + self.init_params[2]*nneurons**(-self.init_params[3]))
            elif self.init_type == 'normal_layerwise':
                self.w.weight.data = torch.normal(self.init_params[0]*nneurons**(-self.init_params[1]), self.init_params[2]*nneurons**(-self.init_params[3]), size=(nneurons,dim))
            elif self.init_type == 'cauchy_layerwise':
                self.w.weight.data = torch.Tensor(np.random.standard_cauchy(size=(nneurons,dim))*self.init_params[0]*nneurons**(-self.init_params[1]) + self.init_params[2]*nneurons**(-self.init_params[3]))
            elif self.init_type == 'xavier_uniform':
                self.w.weight.data = torch.nn.init.xavier_uniform_(self.w.weight.data)
            elif self.init_type == 'xavier_normal':
                self.w.weight.data = torch.nn.init.xavier_normal_(self.w.weight.data)
            elif self.init_type == 'normal_fitting':
                torch.nn.init.xavier_uniform_(self.w.weight.data)
                for i in range(self.output_dim):
                    # for each output neuron, see if the input weights exceed the threshold
                    while self.w.weight.data[:,i].sum() < self.threshold:
                        # if not, reinitialize these input weights until they exceed the threshold
                        self.w.weight.data[:,i] = torch.normal(0.2,1, size=(1,self.input_dim))
            else:
                raise ValueError('init_type {} not implemented'.format(self.init_type))

    def constrain_weights(self, weight):
        '''
        Constrain weights to be positive.
        '''
        return weight*(weight > 0)
    
    def forward(self, x):
        '''
        Forward model.

        Input: tensor of spike times with dimension batchsize x input_dim
        Output: tensor of output spike times with dimension batchsize x output_dim

        The model is solved analytically for the spike times.
        '''
        batchsize, _ = x.size()
        # add input delays if desired
        if self.max_delays != 0:
            input_delay = torch.clamp(self.input_delay, min=0, max=self.max_delays)
            x = x + input_delay

        # sort input spikes for construction of the causal set
        # causal set = all input spikes that caused an output spike
        x, order = torch.sort(x, descending=False) # stable=True needed?
        # reorder weights according to the sorted inputs
        weights = self.w(order)
        # constrain weights to be positive if desired
        if self.positive_weights == True:
            weights = self.constrain_weights(weights)

        tau = self.taus.to(device=x.device, dtype=x.dtype)
        threshold = self.threshold.to(device=x.device, dtype=x.dtype)
        # transform spike times as in Mostafa (2017), but shift the exponent
        # to avoid overflow for large spike times.
        z = x / tau
        z_max = z.max(dim=1, keepdim=True).values
        z_shift = z - z_max
        sp_shifted = torch.exp(z_shift)

        # apply weights to each
        wsumexp = (sp_shifted.unsqueeze(1)*self.mask) @ weights
        wsum = self.mask @ weights
        wsumdiff = wsum-threshold

        # calculate spike times in log-space:
        # log(sum_i w_i exp(t_i/tau) / (sum_i w_i - threshold))
        valid_numerator = wsumexp > 0
        spike_epsilon = self.spike_epsilon.to(device=x.device, dtype=x.dtype)
        valid_denominator = wsumdiff > spike_epsilon
        valid_times = valid_numerator & valid_denominator

        log_wsumexp = torch.where(
            valid_numerator,
            torch.log(torch.where(valid_numerator, wsumexp, torch.ones_like(wsumexp))),
            torch.full_like(wsumexp, float('-inf')),
        )

        valid_log_wquotient = (
            z_max.unsqueeze(-1).expand_as(wsumexp)
            + log_wsumexp
            - torch.log(torch.where(valid_times, wsumdiff, torch.full_like(wsumdiff, 1.0) + spike_epsilon))
        )
        log_wquotient = torch.where(
            valid_times,
            valid_log_wquotient,
            torch.full_like(wsumexp, float('inf')),
        )
        times = tau * log_wquotient

        # apply spike conditions
        # 1. does the input exceed the threshold?
        fmask = valid_times
        # 2. does the next input spike occur before the calculated output spike?
        # if so, we have to include it in our calculation of the  output spike
        next_spike_ok = torch.cat(
            (
                log_wquotient[:, :-1] < z[:, 1:].unsqueeze(-1),
                torch.ones_like(fmask[:, -1:]),
            ),
            dim=1,
        )
        fmask = fmask & next_spike_ok
        # 3. did the neuron already spike? (i.e., only take the first result that fits all criteria)
        # *1 since sort with booleans is not supported on cuda devices
        ffilter, findex = torch.sort(fmask*1., dim=1, descending=True, stable=True)
        batchids = torch.arange(batchsize, device=x.device).unsqueeze(-1).repeat(1,self.output_dim)
        columnids = torch.arange(self.output_dim, device=x.device).repeat(1,batchsize).view(batchsize,self.output_dim)
        # print(batchids, findex[:,0], columnids)
        output_spikes = times[batchids, findex[:,0], columnids]
        # add proxy spike for neurons that did not spike
        output_spikes = torch.where(
            ffilter[:,0] == 1,
            output_spikes,
            self.last_spike.to(device=x.device, dtype=x.dtype),
        )
        if self.debug:
            last_spike = self.last_spike.to(device=x.device, dtype=x.dtype)
            dead_frac = (output_spikes == last_spike).float().mean()
            valid_frac = valid_times.float().mean()
            positive_wsumdiff = wsumdiff[wsumdiff > 0]
            min_positive_wsumdiff = float('nan')
            if positive_wsumdiff.numel() > 0:
                min_positive_wsumdiff = positive_wsumdiff.min().item()
            print(
                'LinearExpIF debug:',
                'dead_frac=', dead_frac.item(),
                'valid_frac=', valid_frac.item(),
                'wsumdiff_min=', wsumdiff.min().item(),
                'wsumdiff_max=', wsumdiff.max().item(),
                'wsumdiff_min_positive=', min_positive_wsumdiff,
                'output_finite=', torch.isfinite(output_spikes).all().item(),
            )
        # the code looks a bit ugly, but all of this is done for all causal sets and a batch of inputs
        # in a vectorized form (instead of looping)

        # add output delays if desired
        if self.max_delays != 0:
            output_delay = torch.clamp(self.output_delay, min=0, max=self.max_delays)
            output_spikes = output_spikes + output_delay

        weight_index = deepcopy(findex[:,0])
        weight_index[ffilter[:,0] !=1] = -1

        return output_spikes, [weight_index, order]
        

    def integrate_model_using_Euler(self, x, t0 = 0, dt = 0.001):
        '''
        Solution for the membrane potential using the analytical solution for u(t) of the ODE.
        From this, spike times can be extracted as well.
        Used as a sanity check for forward call.

        x: input
        t0: start time for integration
        dt: integration time step
        '''
        # get weights in correct shape
        weight = self.w.weight.T
        if self.positive_weights == True:
            weight.data = self.constrain_weights(weight.data)
        return self._integrate_model_using_Euler(x, weight, t0, dt, self.output_dim)

    def _integrate_model_using_Euler(self, x, weight, t0, dt, output_dim):
        '''
        Evaluate membrane potential at every time step (with resolution dt).

        Evalution starts at t0 and ends at t0+2*self.last_spike.
        Uses the analytical solution of the corresponding ODE.

        x: input
        weight: weights
        t0: start time for integration
        dt: integration time step
        output_dim: number of output neurons
        '''
        # add input delays if desired
        if self.max_delays != 0:
            input_delay = torch.clamp(self.input_delay, min=0, max=self.max_delays)
            x += input_delay
        
        # get dimensions
        batch_dim = x.size()[0]

        # some variables for storing
        # note: actual shape adjusts during iteration...
        voltage_trace = []
        spike_times = torch.zeros(output_dim)+self.last_spike
        voltage = torch.zeros(batch_dim, output_dim)

        if 'cuda' in str(weight.device):
            spike_times = spike_times.cuda()
            voltage = voltage.cuda()

        # starting time
        t = t0

        # simply iterate over the analyic solution for the membrane potential.
        # spikes are read out when the membrane potential hits the threshold
        # for the first time.
        voltage_trace.append(deepcopy(voltage.cpu().detach().numpy()))
        while t <= 2*self.last_spike:
            voltage.data = F.linear((1-torch.exp(-(t-x)/self.taus)) * (t > x), weight)
            t += dt
            voltage_trace.append(deepcopy(voltage.cpu().detach().numpy()))

            # spike condition
            # here: hasn't spiked yet + membrane pot. crossed threshold in this time step
            new_spikes = (t-dt-self.last_spike)*(voltage > self.threshold)+self.last_spike
            hasnt_spiked = spike_times == self.last_spike
            spike_times = torch.logical_not(hasnt_spiked)*spike_times + hasnt_spiked*new_spikes

        # add output delays if desired
        if self.max_delays != 0:
            output_delay = torch.clamp(self.output_delay, min=0, max=self.max_delays)
            spike_times += output_delay

        return spike_times, voltage_trace
