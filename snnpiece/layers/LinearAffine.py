import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

class LinearAffine(nn.Module):
    def __init__(self, input_dim, output_dim, lower_bound, upper_bound, seed = None, threshold = 1., 
                 last_spike = None, positive_weights = False, max_delays = 0):
        '''
        Analytical forward model of the Affine SNN model,
        see https://arxiv.org/abs/2404.04549.

        Version for fully connected layer.

        Args:
        input_dim: number of inputs (int)
        output_dim: number of neurons (int)
        seed: for reproducibility (int)
        threshold: spike threshold of output neurons (float)
        last_spike: which spike time to assign when neurons do not fire at all (float)
        positive_weights: whether to constrain weights to be positive (bool)
        max_delays: maximum trainable input and output spike delays (float >= 0). 0: no delay.
        '''
        super().__init__()
        # set seed for reproducibility
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
        # network dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # neuron parameters
        self.threshold = torch.as_tensor(threshold)
        # if a neuron does not spike, this value is used as its spike time
        self.last_spike = torch.as_tensor(last_spike)

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
        self.w.weight.data = torch.normal(0,1, size=(self.w.weight.data.shape))
        # nneurons, dim = self.w.weight.data.shape
        # init_params = [1.2930663 , 0.56592111, 0.85284756, 0.76420202]
        # self.w.weight.data = torch.Tensor(np.random.lognormal(init_params[0]*nneurons**(-init_params[1]), 1, size=(nneurons, dim))*init_params[2]*nneurons**(-init_params[3]))
        # constrain weights to be positive if desired
        if self.positive_weights == True:
            self.w.weight.data = self.constrain_weights(self.w.weight.data)
        if self.positive_weights == False:
            torch.nn.init.xavier_uniform_(self.w.weight.data)
            for i in range(self.output_dim):
                # for each output neuron, see if the input weights exceed the threshold
                while self.w.weight.data[:,i].sum() <= 0:
                    # if not, reinitialize these input weights until they exceed the threshold
                    self.w.weight.data[:,i] = torch.normal(0.2,1, size=(1,self.input_dim))

    def constrain_weights(self, weight):
        '''
        Constrain weights to be positive.
        '''
        return torch.clamp(weight, min=self.lower_bound, max=self.upper_bound)
    
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

        # apply weights to each 
        wsumTimes = (x.unsqueeze(1)*self.mask) @ weights
        wsum = self.mask @ weights

        # calculate spike times
        times = (1+wsumTimes)/(wsum+1e-10) + 1e-10
        
        # apply spike conditions
        # fmask = torch.ones_like(wsum)
        fmask = (wsum>0)
        # 1. does the next input spike occur before the calculated output spike?
        # if so, we have to include it in our calculation of the  output spike
        next_spike_ok = torch.cat(
            (
                times[:, :-1] < x[:, 1:].unsqueeze(-1),
                torch.ones_like(fmask[:, -1:]),
            ),
            dim=1,
        )
        fmask = fmask & next_spike_ok
        # 2. did the neuron already spike? (i.e., only take the first result that fits all criteria)
        # *1 since sort with booleans is not supported on cuda devices
        ffilter, findex = torch.sort(fmask*1., dim=1, descending=True, stable=True)
        batchids = torch.arange(batchsize).unsqueeze(-1).repeat(1,self.output_dim)
        columnids = torch.arange(self.output_dim).repeat(1,batchsize).view(batchsize,self.output_dim)
        # print(batchids, findex[:,0], columnids)
        output_spikes = times[batchids, findex[:,0], columnids]
        # add proxy spike for neurons that did not spike
        output_spikes = torch.where(ffilter[:,0] == 1, output_spikes, self.last_spike)
        # the code looks a bit ugly, but all of this is done for all causal sets and a batch of inputs
        # in a vectorized form (instead of looping)

        # add output delays if desired
        if self.max_delays != 0:
            output_delay = torch.clamp(self.output_delay, min=0, max=self.max_delays)
            output_spikes = output_spikes + output_delay

        return output_spikes, findex[:,0]
        

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
            voltage.data = F.linear(torch.relu((t-x)), self.constrain_weights(weight))
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
