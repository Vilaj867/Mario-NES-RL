import torch
from torch import nn

# initai
class MarioBrain(nn.Module):

    def __init__(self, input, output):
        super.__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.ReLU(),
            nn.flatten(),
            nn.Linear()
        )


    def forward(self):
        pass









    
    
