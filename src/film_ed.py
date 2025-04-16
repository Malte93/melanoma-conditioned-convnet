import argparse
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4
from torchvision.models import EfficientNet_B4_Weights
from film_generator import FiLMGenerator
from copy import deepcopy

class FiLMedNetwork(nn.Module):
    def __init__(self, args : argparse.Namespace) -> None:
        """
        Initializes the FiLM-ed network with its FiLM layers as well as the
        Generator network.

        Parameters
        -------------
            args : argparse.Namespace
                Defined arguments (see train.py).
        
        Further Information
        -------------
            args.film_layer : list
                List of integer values - They determine in which stage (i) the
                film layers shall be active. The corresponding stages with its
                input and output channels are listet below:

                Stage/in_out:    Channels_in     Channels_out
                    stage_one:      in(3)           out(48)
                    stage_two:      in(48)          out(24)
                    stage_three:    in(24)          out(32)
                    stage_four:     in(32)          out(56)
                    stage_five:     in(56)          out(112)
                    stage_six:      in(112)         out(160)
                    stage_seven:    in(160)         out(272)
                    stage_eight:    in(272)         out(448) 
                    stage_nine:     in(448)         out(1792) 
                    classifier:     in(1792)        out(1)
            """
        super().__init__()
        # Init efficientnet-b4 model, film layer and film generator 
        if args.pretrained == True:
            efficientnetb4 = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            efficientnetb4 = efficientnet_b4()
        self.film_layer = args.film_layer
        # When there are active film layers in our FiLM-ed Network we initialize our Generator network.
        # Else: we train our backbone network efficientnet b4 without any contextual information.
        # Which would mean we don't need the FiLM-Generator network.
        if len(self.film_layer) > 0:
            self.film_gen = FiLMGenerator(args)

        # Define film layer sizes (number of output features for linear stages)
        film_layer_sizes = {
            1: 48*2,
            2: 24*2,
            3: 32*2,
            4: 56*2,
            5: 112*2,
            6: 160*2,
            7: 272*2,
            8: 448*2,
            9: 1792*2
        }

        # Initialize those stages (linear layer and FILM) which are defined in film_layer
        for layer_num, out_features in film_layer_sizes.items():
            if layer_num in self.film_layer:
                setattr(self, f'linear_stage_{layer_num}', nn.Linear(in_features=1280*8*8, out_features=out_features, bias=True))
                setattr(self, f'film_{layer_num}', FiLM())

        # Split efficientnet model into submodels w.r.t the stages 1...9 [1]
        for stage_num in range(1, 10):
            setattr(self, f'stage_{stage_num}', deepcopy(efficientnet_b4.features[stage_num]))
        self.avg = deepcopy(efficientnetb4.avgpool)
        efficientnetb4.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1792, out_features=1300, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1300, out_features=650, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=650, out_features=1, bias=True)
        )
        dropout_idx = [3, 6]
        batchnorm_config = [(2, 1300), (5, 650)] # [(idx_1, num_features_1), ...]
        classification_layers = list(efficientnetb4.classifier)
        if args.dropout_active:
            for idx in dropout_idx:
                classification_layers.insert(idx, nn.Dropout(p=0.3))
        if args.dropout_active and args.batchnorm_active:
            batchnorm_config[1] = (6, 650)
        if args.batchnorm_active:
            for idx, num_features in batchnorm_config:
                classification_layers.insert(idx, nn.BatchNorm1d(num_features=num_features))     
        efficientnetb4.classifier = nn.Sequential(*classification_layers)
        self.classifier = deepcopy(efficientnetb4.classifier)

    def forward(self, img : torch.tensor, auxiliary_img : torch.tensor) -> torch.tensor:
        """
        Takes in a tuple x (img, auxiliary_img) which consists of an image which
        shall be classified and a second image which serves as conditional information. 
        This function calculates the FiLM-ed network output and returns it
        for further processing.

        Parameters
        -------------
            img : torch.tensor
                The image which shall be classified as malignant or benign.
            auxiliary_img : torch.tensor
                The auxiliiary input which serves as contextual information
                (same patient as from img).

        Returns
        -------------
            out : tensor
                Scaling and shifting parameter (gamma and beta).
        """
        gen_out = auxiliary_img
        if hasattr(self, 'film_gen'):
            gen_out = self.film_gen(auxiliary_img)
        
        out = img 
        for i in range(1, 10):
            stage = getattr(self, f'stage_{i}', None)
            if stage:
                out = stage(out)

            if i in self.film_layer:
                linear_stage = getattr(self, f'linear_stage_{i}', None)
                film_layer = getattr(self, f'film_{i}', None)
                linear_stage_out = linear_stage(gen_out)
                out = film_layer(out, linear_stage_out)                    

        out = self.avg(out)
        out = torch.flatten(out, start_dim=1)
        out = self.classifier(out)

        return out

class FiLM(nn.Module):
    """
    Feature wise modulated layer. Calculates the gamma (scaling) and beta (shifting)
    parameters.  
    """
    def forward(self, x : torch.tensor, auxiliary : torch.tensor) -> torch.tensor:
        """
        This function calculates the the beta and gamma parameters with the aid 
        of the already transformed image (which shall be classified) and the
        auxiliary image (which serves as the conditional information).
        
        Parameters
        -------------
            x : torch.tensor
                The transformed (image) output which shall be classified.
            auxiliary_img : torch.tensor
                The transformed auxiliary input which serves as contextual
                information (same patient as from x).
        
        Returns
        -------------
            (gammas * x + betas) : torch.tensor
        """
        betas, gammas = torch.split(auxiliary, auxiliary.size(1) // 2, dim=1)
        betas = betas.unsqueeze(-1).unsqueeze(-1)
        gammas = gammas.unsqueeze(-1).unsqueeze(-1)

        return (gammas * x + betas)