from transformers.models.clip.modeling_clip import CLIPTextTransformer
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import CLIPTextModel

class AdaptedCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, *args, **kwargs):
        super(AdaptedCLIPTextTransformer, self).__init__(*args, **kwargs)

        # Define the adaptation layer within CLIPTextTransformer
        self.adaptation_layer = nn.Linear(512, 768).to(dtype=torch.float16)

    def forward(self, input_ids, **kwargs):
        outputs = super().forward(input_ids, **kwargs)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = last_hidden_state.to(dtype=torch.float16)

        # Apply the adaptation layer
        adapted_outputs = F.linear(last_hidden_state, self.adaptation_layer.weight, self.adaptation_layer.bias)

        return adapted_outputs

class ModifiedPLIPEncoder(CLIPTextModel):
    def __init__(self, *args, **kwargs):
        super(ModifiedPLIPEncoder, self).__init__(*args, **kwargs)

        # Replace the text model with the adapted version
        self.text_model = AdaptedCLIPTextTransformer(*args, **kwargs)