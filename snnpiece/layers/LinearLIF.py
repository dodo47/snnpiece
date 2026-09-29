import torch
import torch.nn.functional as F
from copy import deepcopy

from .LinearExpIF import LinearExpIF


class LinearLIF(LinearExpIF):
    def __init__(
        self,
        input_dim,
        output_dim,
        seed=None,
        threshold=1.0,
        taus=0.1,
        taum=None,
        last_spike=None,
        positive_weights=False,
        max_delays=0,
        init_type='xavier_normal',
        init_params=None,
        spike_epsilon=1e-3,
    ):
        """
        Analytical forward model of the current-based LIF neuron with exponential synapses
        in the special case tau_m = 2 * tau_s.

        The implementation mirrors LinearExpIF as closely as possible: inputs are sorted,
        all prefix causal sets are evaluated in parallel, and the first valid causal set is
        selected. The only substantial difference is the closed-form spike-time equation.

        Note: Goltz et al, this would be g_l = 1/2
        """
        if taum is None:
            taum = 2 * taus
        if abs(float(taum) - 2.0 * float(taus)) > 1e-8:
            raise ValueError('LinearLIF currently only supports the special case taum = 2 * taus')

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            seed=seed,
            threshold=threshold,
            taus=taus,
            last_spike=last_spike,
            positive_weights=positive_weights,
            max_delays=max_delays,
            init_type=init_type,
            init_params=init_params,
            spike_epsilon=spike_epsilon,
        )
        self.taum = torch.as_tensor(taum)

    def forward(self, x):
        batchsize, _ = x.size()
        if self.max_delays != 0:
            input_delay = torch.clamp(self.input_delay, min=0, max=self.max_delays)
            x = x + input_delay

        x, order = torch.sort(x, descending=False)
        weights = self.w(order)
        if self.positive_weights is True:
            weights = self.constrain_weights(weights)

        tau = self.taus.to(device=x.device, dtype=x.dtype)
        threshold = self.threshold.to(device=x.device, dtype=x.dtype)
        z = x / tau
        z_half = z / 2

        # For each prefix causal set k, shift by its own latest input x_k.
        # This mirrors LinearExpIF's stable exponentials without letting future
        # inputs suppress earlier prefixes numerically.
        rel_full = z.unsqueeze(1) - z.unsqueeze(2)
        rel_half = z_half.unsqueeze(1) - z_half.unsqueeze(2)
        mask = self.mask.unsqueeze(0).bool()
        rel_full = rel_full.masked_fill(~mask, float('-inf'))
        rel_half = rel_half.masked_fill(~mask, float('-inf'))
        sp_prefix_shifted = torch.exp(rel_full)
        sp_half_prefix_shifted = torch.exp(rel_half)

        a_shifted = sp_prefix_shifted @ weights
        b_shifted = sp_half_prefix_shifted @ weights

        # Solve the special-case tau_m = 2 * tau_s LIF spike-time equation in
        # shifted space. If prefix k is shifted by x_k, the physical spike time
        # is x_k + 2 * tau * log(local_ratio).
        discriminant = b_shifted.pow(2) - 2 * a_shifted * threshold
        valid_disc = discriminant >= 0
        sqrt_disc = torch.where(
            valid_disc,
            torch.sqrt(torch.where(valid_disc, discriminant, torch.zeros_like(discriminant))),
            torch.zeros_like(discriminant),
        )

        denom = b_shifted + sqrt_disc
        spike_epsilon = self.spike_epsilon.to(device=x.device, dtype=x.dtype)
        valid_denom = denom > spike_epsilon
        valid_ratio = valid_disc & valid_denom & (a_shifted > 0)
        ratio = torch.where(
            valid_ratio,
            (2 * a_shifted) / torch.where(valid_ratio, denom, torch.ones_like(denom)),
            torch.full_like(denom, float('inf')),
        )

        valid_times = valid_ratio & torch.isfinite(ratio) & (ratio > 0)
        log_ratio = torch.where(
            valid_times,
            torch.log(torch.where(valid_times, ratio, torch.ones_like(ratio))),
            torch.full_like(ratio, float('inf')),
        )
        times = torch.where(
            valid_times,
            x.unsqueeze(-1) + 2 * tau * log_ratio,
            torch.full_like(ratio, float('inf')),
        )

        fmask = valid_times
        after_latest = times >= x.unsqueeze(-1)
        next_spike_ok = torch.cat(
            (
                times[:, :-1] <= x[:, 1:].unsqueeze(-1),
                torch.ones_like(fmask[:, -1:]),
            ),
            dim=1,
        )
        fmask = fmask & after_latest & next_spike_ok
        ffilter, findex = torch.sort(fmask * 1.0, dim=1, descending=True, stable=True)

        batchids = torch.arange(batchsize, device=x.device).unsqueeze(-1).repeat(1, self.output_dim)
        columnids = torch.arange(self.output_dim, device=x.device).repeat(1, batchsize).view(batchsize, self.output_dim)
        output_spikes = times[batchids, findex[:, 0], columnids]
        output_spikes = torch.where(
            ffilter[:, 0] == 1,
            output_spikes,
            self.last_spike.to(device=x.device, dtype=x.dtype),
        )

        if self.max_delays != 0:
            output_delay = torch.clamp(self.output_delay, min=0, max=self.max_delays)
            output_spikes = output_spikes + output_delay

        weight_index = deepcopy(findex[:, 0])
        weight_index[ffilter[:, 0] != 1] = -1

        return output_spikes, [weight_index, order]

    def _integrate_model_using_Euler(self, x, weight, t0, dt, output_dim):
        if self.max_delays != 0:
            input_delay = torch.clamp(self.input_delay, min=0, max=self.max_delays)
            x += input_delay

        batch_dim = x.size()[0]
        voltage_trace = []
        spike_times = torch.zeros(output_dim) + self.last_spike
        voltage = torch.zeros(batch_dim, output_dim)

        if 'cuda' in str(weight.device):
            spike_times = spike_times.cuda()
            voltage = voltage.cuda()

        t = t0
        voltage_trace.append(deepcopy(voltage.cpu().detach().numpy()))
        while t <= 2 * self.last_spike:
            psp = 2 * (
                torch.exp(-(t - x) / (2 * self.taus)) - torch.exp(-(t - x) / self.taus)
            ) * (t > x)
            voltage.data = F.linear(psp, weight)
            t += dt
            voltage_trace.append(deepcopy(voltage.cpu().detach().numpy()))

            new_spikes = (t - dt - self.last_spike) * (voltage > self.threshold) + self.last_spike
            hasnt_spiked = spike_times == self.last_spike
            spike_times = torch.logical_not(hasnt_spiked) * spike_times + hasnt_spiked * new_spikes

        if self.max_delays != 0:
            output_delay = torch.clamp(self.output_delay, min=0, max=self.max_delays)
            spike_times += output_delay
            

        return spike_times, voltage_trace
