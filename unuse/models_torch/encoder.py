import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.use_conv = kwargs['enc_use_conv']
        self.use_mlp = kwargs['enc_use_mlp']
        
        # Conv and MLP are sequential, check if conv_output dim = mlp_input dim
        if self.use_conv and self.use_mlp:
            assert kwargs['enc_out_channel'] == kwargs['enc_input_dim']
        
        
        self.in_channel = kwargs['enc_in_channel'] # 1
        self.conv_dims = kwargs['enc_conv_dims'] # N - 1 
        self.kernel_sizes = kwargs['enc_kernel_sizes'] # N
        self.strides = kwargs['enc_strides'] # N
        self.out_channel = kwargs['enc_out_channel'] # 1
        
        
        self.conv_layers = []
        self.conv_layers.append(nn.Conv2d(in_channels=self.in_channels, 
                                    out_channels=self.conv_dims[0], 
                                    kernel_size=self.kernel_sizes[0], 
                                    stride=self.strides[0]))
        
        for layer_ind in range(1, len(self.conv_dims)+1):
            self.conv_layers.append(nn.Conv2d(in_channels=self.conv_dims[layer_ind-1], 
                                        out_channels=self.conv_dims[layer_ind], 
                                        kernel_size=self.kernel_sizes[layer_ind], 
                                        stride=self.strides[layer_ind]))
            
        self.conv_layers.append(nn.Conv2d(in_channels=self.conv_dims[-1], 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.kernel_sizes[-1], 
                                    stride=self.strides[-1]))
        
        self.conv_enc = nn.Sequential(*self.conv_layers)

        
        self.input_dim = kwargs['enc_input_dim']
        self.mlp_dims = kwargs['enc_mlp_dims']
        self.output_dim = kwargs['enc_output_dims']
        
        self.mlp_layers = []
        self.mlp_layers.append(nn.Linear(in_features=self.input_dim,
                                       out_features=self.mlp_dims[0]))
        self.mlp_layers.append(self.activ_fn)
        
        for layer_ind in range(1, len(self.mlp_dims)+1):
            self.mlp_layers.append(nn.Linear(in_features=self.mlp_dims[layer_ind-1],
                                    out_features=self.mlp_dims[layer_ind]))
            self.mlp_layers.append(self.activ_fn)
            
        self.mlp_layers.append(nn.Linear(in_features=self.mlp_dims[-1],
                                       out_features=self.output_dim))
        
        self.mlp_enc = nn.Sequential(*self.mlp_layers)
        
               
    def forward(self, obs):
        features = obs
        
        if self.use_conv:
            features = self.conv_enc(features)
        
        if self.use_mlp:
            features = self.mlp_enc(features)
        
        return features