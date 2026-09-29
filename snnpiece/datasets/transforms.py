import torch

class VecToLatencies(torch.nn.Module):
    """
    Turn pixel values to latencies. Expects values between 0 and 1.
    Simply linearly scales, i.e., pixel values of 1 are mapped to 0, 0 to 1, etc.
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return 1-tensor