import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)

def ResidualBlock(in_channels, out_channels, kernel_size, dilation):
    return Residual(nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding='same')
    ))

class SpliceAI_10k(nn.Module):
    S = 10000

    def __init__(self, in_channels = 4, out_channels = 3, n_embed = 32):
        super().__init__()

        self.receptive_field = 5000

        self.conv1 = nn.Conv1d(in_channels, n_embed, 1, dilation=1, padding='same')
        
        self.res_conv = nn.ModuleList( [ nn.Conv1d(n_embed, n_embed, 1, dilation=1, padding='same') for _ in range(4) ] )

        kernel_sizes = [ 11, 11, 21, 41 ]
        dilations = [ 1, 4, 10, 25 ]
        
        self.blocks = nn.ModuleList( [ nn.Sequential( *[ ResidualBlock(n_embed, n_embed, kernel_size, dilation) for _ in range(4) ] ) for (kernel_size, dilation) in zip(kernel_sizes, dilations) ] )

        self.conv_last = nn.Conv1d(n_embed, out_channels, 1, dilation=1, padding='same') # 3 channels for donor, acceptor, neither
    
    def forward(self, x):
        x = self.conv1(x)

        for i in range(4): 
            res_conv = self.res_conv[i](x)
            detour = res_conv if (i==0) else (detour + res_conv)
            x = self.blocks[i](x)
        
        x += detour
        
        x = self.conv_last(x)

        return x[:, :, self.receptive_field:-self.receptive_field ]

if __name__ == '__main__':

    x = torch.randn([16, 4, 10000 + 5000])
    model = SpliceAI_10k(out_channels = 1)
    print(model(x).shape)