import torch

class ttfsLoss2021(torch.nn.modules.loss._Loss):
    '''
    Loss function used in https://arxiv.org/abs/1912.11443.
    '''
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_spiketime = prediction[torch.arange(target.size()[0]), target].unsqueeze(-1)
        spikediff_to_target = (target_spiketime-prediction)/self.scaling
        loss = torch.logsumexp(spikediff_to_target, dim=1).mean()
        return loss
