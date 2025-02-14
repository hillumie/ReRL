import argparse
import tqdm
import wandb
import re
from omegaconf import OmegaConf
from pathlib import Path
import hydra
from rich.table import Table
from rich.console import Console
        
    
def print_cfgs(cfgs):
    """
    Print configuration values to the terminal in Rich table
    """
    cfgs_table = Table(title='Configuration settings')
    cfgs_table.add_column('Configurations', style='orange1')
    cfgs_table.add_column('Values', style='medium_spring_green')
    for k in cfgs.keys():
        cfgs_table.add_row(str(k), str(cfgs[k]))
    
    console = Console()
    console.print(cfgs_table)