import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder

class BaseQNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Base Q Network 
        Arguments:
        
        """
        super(BaseQNet, self).__init__(**kwargs)
        
        self.state_dim = kwargs['state_dim']
        self.hidden_dims = kwargs['hidden_dims']
        self.activ_fn = kwargs['activ_fn']
        self.num_action = kwargs['num_action']
        
        assert kwargs['state_dim'] == kwargs['enc_output_dim']
        
        self.q_layers = []
        self.q_layers.append(nn.Linear(in_features=self.state_dim,
                                       out_features=self.hidden_dims[0]))
        self.q_layers.append(self.activ_fn)
        
        for layer_ind in range(1, len(self.hidden_dims)+1):
            self.q_layers.append(nn.Linear(in_features=self.hidden_dims[layer_ind],
                                           out_features=self.hidden_dims[layer_ind]))
            self.q_layers.append(self.activ_fn)
            
        self.q_layers.append(nn.Linear(in_features=self.hidden_dims[-1],
                                       out_features=self.num_action))
        
        self.q_net = nn.Sequential(*self.q_layers)
    
    def forward(self, state):
        return self.q_net(state)   
            
            