import argparse
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights
from copy import deepcopy

class FiLMGenerator(nn.Module):
    def __init__(self, args : argparse.Namespace) -> None:
        """
        Initializes the Generator for the feature-wise-linear modulation network (FiLM-ed network).
        The backbone net for the Generator is an Efficientnet b0 without its classification layer.
        
        Parameters
        -------------
            args : argparse.Namespace
                Defined arguments (see train.py).
        """
        super(FiLMGenerator, self).__init__()
        # Initialize the effnet-b0
        if args.pretrained == True:
            efficientnetb0 = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            efficientnetb0 = efficientnet_b0()
        # Take only the non-classification layers from the efficientnet
        self.gen = deepcopy(efficientnetb0.features)

    def forward(self, auxiliary_input: torch.tensor) -> torch.tensor:
        """
        Represents the forward path for the FiLM-Generator.

        Parameters
        -------------
            auxiliary_input : torch.tensor 
                Conditional information (in this case a lesion image from the
                same patient as from the lesion image which shall be classified)
        
        Returns
        -------------
            out : torch.tensor
                The calculated logits from the FiLM-Generator
            """
        out = self.gen(auxiliary_input)
        out = torch.flatten(out, start_dim=1)

        return out