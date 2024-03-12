import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import transcript_data
from mamba_ssm import Mamba

def my_bce_loss(seq_mask, mask, logits, one_hot):
    seq_eval_mask = seq_mask & ~mask[:,0,:] 
    seq_out = logits.permute(0, 2, 1)
    seq_out_norm = seq_out - seq_out.logsumexp(2, keepdims = True)
    one_hot_t = one_hot.permute(0, 2, 1) # no op
    return - (one_hot_t[ seq_eval_mask ] * seq_out_norm[ seq_eval_mask ]).sum() / seq_eval_mask.sum() 


class Chomp1d(nn.Module): 
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] # .contiguous() # needed? 

class MambaBlock(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.mixer = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=n_embed, # Model dimension d_model
            d_state=16,  # SSM state expansion factor (this is the default)
            d_conv=4,    # Local convolution width (this is the default)
            expand=2,    # Block expansion factor (this is the default)
        ) # .to("cuda")
        self.ln = nn.LayerNorm(n_embed)

    def forward(self, x): # x must be batch x time x channels
        return self.ln(x + self.mixer(x))

class BidirMambaBlock(nn.Module):
    def __init__(self, n_embed, weight_tie = False) -> None:
        super().__init__()
        self.mixer = Mamba( d_model=n_embed ) 
        self.mixer_back = Mamba( d_model=n_embed ) 
        self.ln = nn.LayerNorm(n_embed)

        if weight_tie:  # Tie in and out projections (where most of param count lies) (from Caduceus)
            self.mixer_back.in_proj.weight = self.mixer.in_proj.weight
            self.mixer_back.in_proj.bias = self.mixer.in_proj.bias
            self.mixer_back.out_proj.weight = self.mixer.out_proj.weight
            self.mixer_back.out_proj.bias = self.mixer.out_proj.bias

    def forward(self, x): # x must be batch x time x channels
        mix_flip = self.mixer_back(x.flip(1))
        return self.ln(x + self.mixer(x) + mix_flip.flip(1))

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout1 = nn.Dropout1d(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout2 = nn.Dropout1d(dropout)

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
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, stride=1, dropout=0.0):
        super().__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout1 = nn.Dropout1d(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout2 = nn.Dropout1d(dropout)

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

class TemporalConvNet(nn.Module):
    def __init__(self, num_channels, kernel_size=2, dropout=0.0):
        super().__init__()

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

            channels_in = 5 if i==0 else num_channels[i-1] # *2
            if i==0: 
                self.in_proj = nn.Linear(channels_in, num_channels[i]) # or could use nn.Embedding? maybe equivalent...
            #TemporalBlock(channels_in, num_channels[i], kernel_size, dilation, rf, dropout=dropout)
            self.causal_convs.append( MambaBlock(num_channels[i])  )
            channels_in = 4 if i==0 else num_channels[i-1]
            self.noncausal_convs.append( ResBlock(channels_in, num_channels[i], kernel_size, dilation, 0, dropout=dropout )) # int(padding / 2)
        
        #self.causal_convs.append( nn.Linear(num_channels[num_levels-1]*2, 1) )
        self.out_proj = nn.Conv1d(num_channels[num_levels-1], 1, 1, stride=1, padding=0, dilation=1) 
        self.receptive_field = sum(self.receptive_fields)
    
    def forward(self, causal_net, noncausal_net):
        
        causal_net = self.pad(self.chomp(causal_net)) # shift by one
        num_layers = len(self.noncausal_convs)
        nbatch = causal_net.shape[0]
        for i in range(num_layers):

            causal_len = causal_net.shape[2]
            noncausal_len = noncausal_net.shape[2]
            to_trim = int((noncausal_len - causal_len)/2)
            noncausal_net_trimmed = noncausal_net[:,:,to_trim:-to_trim]

            # B x C x L
            concat = torch.cat( [causal_net, noncausal_net_trimmed.expand(nbatch,-1,-1)], axis=1) if i==0 else (causal_net + noncausal_net_trimmed)
            concat = concat.permute(0,2,1) # now B x L x C
            if i==0: 
                concat = self.in_proj(concat)
            causal_net = self.causal_convs[i](concat)
            causal_net = causal_net.permute(0,2,1) # back to B x C x L
            
            noncausal_net = self.noncausal_convs[i](noncausal_net)

        assert(causal_net.shape[2] == noncausal_net.shape[2]) # rf has all been used up
        #concat = torch.cat([causal_net, noncausal_net.expand(nbatch,-1,-1)], axis=1)
        concat = causal_net + noncausal_net_trimmed
        return(self.out_proj(concat))


class ConvNet(nn.Module):
    def __init__(self, num_channels, kernel_size=2, dropout=0.0):
        super(ConvNet, self).__init__()

        num_levels = len(num_channels)

        self.noncausal_convs = nn.ModuleList()

        self.receptive_field = 0
        self.receptive_fields = []

        for i in range(num_levels):
            dilation = 2 ** i

            rf = int( (kernel_size-1) * dilation )
            self.receptive_fields.append(rf)

            assert(rf % 2 == 0)

            channels_in = 5 if i==0 else num_channels[i-1] # *2
            if i==0: 
                self.in_proj = nn.Linear(channels_in, num_channels[i]) # or could use nn.Embedding? maybe equivalent...

            channels_in = 4 if i==0 else num_channels[i-1]
            self.noncausal_convs.append( ResBlock(channels_in, num_channels[i], kernel_size, dilation, 0, dropout=dropout )) # int(padding / 2)
        
        #self.causal_convs.append( nn.Linear(num_channels[num_levels-1]*2, 1) )
        self.out_proj = nn.Conv1d(num_channels[num_levels-1], 1, 1, stride=1, padding=0, dilation=1) 
        self.receptive_field = sum(self.receptive_fields)
    
    def forward(self, causal_net, noncausal_net):
        """ Causal net just ignored """
        
        num_layers = len(self.noncausal_convs)
        
        for i in range(num_layers):
            noncausal_net = self.noncausal_convs[i](noncausal_net)
            
        assert(causal_net.shape[2] == noncausal_net.shape[2]) # rf has all been used up
        return(self.out_proj(noncausal_net))

class MambaNet(nn.Module):
    
    def __init__(self, vocab_size, input_channels, n_embed, n_layers, pos_embedding = False, bidir = False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed) if pos_embedding else None
        self.in_proj = nn.Linear(input_channels, n_embed) 
        block = BidirMambaBlock if bidir else MambaBlock
        self.blocks = nn.Sequential(*[block(n_embed) for _ in range(n_layers)])
        self.lm_head = nn.Linear(n_embed, vocab_size) # TODO: tie to embeddings
        self.out_proj = nn.Linear(n_embed, input_channels)
        
    def forward(self, seq, input):
        """
        seq: DNA sequence, integer encoding
        input: other stuff (continuous in general), e.g. is_exonic
        """
        B,T = seq.shape
        x = self.token_embedding_table(seq) + self.in_proj(input) # (B,T,C_e)
        if self.position_embedding_table:
          x += self.position_embedding_table(torch.arange(T,device=idx.device)) # (T,C_e)
        x = self.blocks(x) # (B,T,C_e)
        return self.lm_head(x), self.out_proj(x)

class MambaOneHotNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, n_embed, n_layers, receptive_field, bidir = False):
        super().__init__()
        self.receptive_field = receptive_field
        self.in_proj = nn.Linear(in_channels, n_embed) # better off with a convolution? 
        block = BidirMambaBlock if bidir else MambaBlock
        self.blocks = nn.Sequential(*[block(n_embed) for _ in range(n_layers)])
        self.out_proj = nn.Linear(n_embed, out_channels)
        
    def forward(self, x):
        """
        seq: DNA sequence, integer encoding
        input: other stuff (continuous in general), e.g. is_exonic
        """
        x = x.permute(0,2,1)
        x = self.in_proj(x) # (B,T,C_e)
        x = self.blocks(x) # (B,T,C_e)
        return self.out_proj(x).permute(0,2,1)[:, :, self.receptive_field:-self.receptive_field ]

