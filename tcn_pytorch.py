#%% 
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import transcript_data

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] # .contiguous() # needed? 


class SpatialDropout1D(nn.Module):

    def __init__(self, p):
        super(SpatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)   # back to [batch, time, channels]
        return(x)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, stride=1, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout1 = SpatialDropout1D(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout2 = SpatialDropout1D(dropout)

        self.net = nn.Sequential(self.conv1, Chomp1d(padding), nn.ELU(), self.dropout1,
                                 self.conv2, Chomp1d(padding), nn.ELU(), self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class ResBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, stride=1, dropout=0.2):
        super(ResBlock, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout1 = SpatialDropout1D(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout2 = SpatialDropout1D(dropout)

        self.net = nn.Sequential(self.conv1, nn.ELU(), self.dropout1,
                                 self.conv2, nn.ELU(), self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        if self.padding == 0: 
            to_trim = (self.kernel_size - 1) * self.dilation
            assert(to_trim % 2 == 0)
            #to_trim /= 2 # don't do this because of two convs
            to_trim = int(to_trim)
            x = x[:,:,to_trim:-to_trim]
    
        if not self.downsample is None: 
            x = self.downsample(x)
        
        return F.relu(out + x)

#%%
class TemporalConvNet(nn.Module):
    def __init__(self, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()

        num_levels = len(num_channels)

        self.pad = nn.ConstantPad1d((1,0), 0.0)
        self.chomp = Chomp1d(1)

        self.causal_convs = nn.ModuleList()
        self.noncausal_convs = nn.ModuleList()

        self.receptive_field = 0
        self.receptive_fields = []

        for i in range(num_levels):
            dilation = 2 ** i

            rf = int( (kernel_size-1) * dilation )
            self.receptive_fields.append(rf)

            assert(rf % 2 == 0)

            channels_in = 5 if i==0 else num_channels[i-1]*2
            self.causal_convs.append( TemporalBlock(channels_in, num_channels[i], kernel_size, dilation, rf) )
            channels_in = 4 if i==0 else num_channels[i-1]
            self.noncausal_convs.append( ResBlock(channels_in, num_channels[i], kernel_size, dilation, 0 )) # int(padding / 2)
            
        #self.concats.append( keras.layers.Concatenate() )
        self.causal_convs.append( nn.Conv1d(num_channels[num_levels-1]*2, 1, 1, stride=1, padding=0, dilation=dilation) )
        self.receptive_field = sum(self.receptive_fields)
    
    def forward(self, causal_net, noncausal_net):
        
        causal_net = self.pad(self.chomp(causal_net)) # shift by one
        num_layers = len(self.noncausal_convs)
        nbatch = causal_net.shape[0]
        for i in range(num_layers):

            #causal_len = causal_net.shape[2]
            #noncausal_len = noncausal_net.shape[2]
            #if noncausal_len > causal_len: 
            #to_trim = int((noncausal_len - causal_len)/2)
            to_trim = int(self.receptive_fields[i]/2)
            noncausal_net_trimmed = noncausal_net[:,:,to_trim:-to_trim]
            #else:
            #    noncausal_net_trimmed = noncausal_net

            concat = torch.cat( [causal_net, noncausal_net_trimmed.expand(nbatch,-1,-1)], axis=1)
            causal_net = self.causal_convs[i](concat)
            noncausal_net = self.noncausal_convs[i](noncausal_net)

        assert(causal_net.shape[2] == noncausal_net.shape[2]) # rf has all been used up
        concat = torch.cat([causal_net, noncausal_net.expand(nbatch,-1,-1)], axis=1)
        return(self.causal_convs[num_layers](concat))

model = TemporalConvNet([32] * 10, kernel_size=7)
print(model)
#%%
import time

optimizer = torch.optim.Adam(model.parameters())

losses = []
start_time = time.time()

model.train()
for ((is_exon, one_hot), is_exon) in transcript_data.get_gene(model.receptive_field): 
    is_exon = np.swapaxes(is_exon, 1, 2)
    one_hot = np.swapaxes(one_hot, 0, 1)
    #one_hot = np.pad(one_hot, ((0,0),(0,0),(rf,rf)), "constant")
    one_hot = one_hot[np.newaxis,:,:]
    is_exon = torch.from_numpy(is_exon)
    #is_exon_y = torch.from_numpy(one_hot.astype(bool))
    one_hot = torch.from_numpy(one_hot)
    optimizer.zero_grad()
    output = model(is_exon, one_hot)
    loss = F.binary_cross_entropy_with_logits(is_exon, output)
    loss.backward()
    optimizer.step()
    losses.append( loss.item() )

    acc = (is_exon > 0.5).eq( output > 0. ).float().mean()
    print("%f %f" % (np.mean(losses), acc))
