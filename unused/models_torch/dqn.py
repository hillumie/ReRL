import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.q_net import BaseQNet
from memory import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Initialize Deep Q Network 
        Arguments:
        
        """
        super(DQN, self).__init__(**kwargs)

        self.enc = Encoder(kwargs=kwargs)
        self.q_net = BaseQNet(kwargs=kwargs)
    
    def forward(self, obs):
        x = self.enc(obs)
        x = self.q_net(x)
        
        return x
        
            