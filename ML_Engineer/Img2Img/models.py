import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    """A single conv layer to learn a 2D image filter.
    Args:
        num_out_channels: (1 or 2) depending on whether learning both filters or single.
        combine: True if output channels should be combined, or False if output channels
                 should be intact
    """
    
    def __init__(self, num_out_channels=1, combine=True):
        super().__init__()
        self.num_out_channels = num_out_channels
        self.combine = combine
        self.conv_layer = nn.Conv2d(1, num_out_channels, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, x):
        y = self.conv_layer(x) # B, C, H, W
        if self.combine and self.num_out_channels > 1:
            y = torch.sqrt(
                torch.sum(torch.pow(y, 2), axis=1))  # B, H, W
            y = y[:, None, :, :] # introduce a new axis.
        return y  # B, C, H, W
    

class SobelFilterEfficient(nn.Module):
    """A single conv layer to learn a 2D image filter.
    Args:
        num_out_channels: (1 or 2) depending on whether learning both filters or single.
        combine: True if output channels should be combined, or False if output channels
                 should be intact
    """
    
    def __init__(self, num_out_channels=1, combine=True):
        super().__init__()
        self.num_out_channels = num_out_channels
        self.combine = combine
        self.vec1 = torch.nn.Parameter(torch.rand((1, 3), requires_grad=True))
        self.vec2 = torch.nn.Parameter(torch.rand((1, 3), requires_grad=True))
        
    @property
    def kernel(self):
        if self.combine and self.num_out_channels == 2:
            weight = torch.cat(
                ((self.vec1.T @ self.vec2)[None]
                (self.vec2.T @ self.vec1)[None]), axis=0
            )[None]
        else:
            weight = torch.matmul(self.vec1.T, self.vec2).reshape(1, 1, 3, 3)
        return weight

    def forward(self, x):
        if self.combine and self.num_out_channels == 2:
            weight = torch.cat(
                ((self.vec1.T @ self.vec2).unsqueeze(0),
                (self.vec2.T @ self.vec1).unsqueeze(0)), axis=0
            ).unsqueeze(0)
        else:
            weight = torch.matmul(self.vec1.T, self.vec2).reshape(1, 1, 3, 3)
        
        # flip because conv is implemented as cross-correlation in torch.
        weight = torch.flip(weight, [2, 3])    
        y = F.conv2d(x, weight, padding=1)  # B, C, H, W
            
        if self.combine and self.num_out_channels > 1:
            y = torch.sqrt(
                torch.sum(torch.pow(y, 2), axis=1)) # B, H, W
            y = y.unsqueeze(1) # introduce a new axis.
        
        return y
            

class GeneralFilter(SobelFilter):
    """A single conv layer to learn a ND image filter.
    Args:
        num_out_channels: (1 or 2) depending on whether learning both filters or single.
        combine: True if output channels should be combined, or False if output channels
                 should be intact
        kernel_size: int indicating the size of the square kernel.
    """
    
    def __init__(self, num_out_channels=1, combine=True, kernel_size=3):
        super().__init__(num_out_channels, combine)
        self.kernel_size = kernel_size
        self.conv_layer = nn.Conv2d(1, num_out_channels, kernel_size=kernel_size, stride=1, padding='same')
