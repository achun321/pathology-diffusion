from transformers import CLIPModel
import torch.nn as nn
import torch.nn.functional as F
import torch

class ModifiedPLIPModel(nn.Module):
    def __init__(self, plip_encoder, input_dim, output_dim):
        super(ModifiedPLIPModel, self).__init__()
        self.plip_encoder = plip_encoder
        self.adaptation_layer = nn.Linear(input_dim, output_dim).to(dtype=torch.float16)
        
    def forward(self, x):
        x = self.plip_encoder(x)[0]
        x = F.linear(x, self.adaptation_layer.weight, self.adaptation_layer.bias)
        return x