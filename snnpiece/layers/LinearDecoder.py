import torch
import torch.nn as nn

class LinearDecoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, seed = None, trainable = False):
        """
        Initialize a linear decoder with an optional switch to turn off plasticity.

        Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        trainable (bool): whether to allow plasticity in the decoder. Defaults to False
        """
        super().__init__()
        # set seed for reproducibility
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)

        # decoder is a simple linear layer
        self.l1 = nn.Linear(in_dim, out_dim, bias=True)

        # turn off plasticity if desired
        if trainable == False:
            self.l1.weight.requires_grad = False
            self.l1.bias.requires_grad = False
            # initialize weights using normal distribution
            self.l1.weight.data = torch.normal(0, 1, size=self.l1.weight.size())

    def forward(self, x):
        # Avoid in-place writes so upstream autograd state remains valid.
        x = torch.where(x > 4.5, torch.zeros_like(x), x)
        return self.l1(x), None
