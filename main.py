import numpy as np
import jax
import jax.numpy as jnp
import flax.linen  as nn
from flax.training.train_state import TrainState
from flax import struct
import chex
import optax


import gymnasium as gym
import argparse
import tqdm
import wandb
from omegaconf import OmegaConf
import logging
import hydra
# from termcolor import colored
import rich

import warnings
warnings.filterwarnings('ignore')
import time
from agent import Agent

# logging.basicConfig()
# logging.getLogger().setLevel(logging.ERROR)

logger = logging.getLogger("jax._src")
logger.setLevel(logging.ERROR)

# from utils.common_utils import set_seed


@hydra.main(version_base=None, config_path='./configs', config_name='default')
def main(config):
    """Whole running section"""
    
    cfg = OmegaConf.to_container(config)
    
    agent = Agent(cfg)
    
    agent.run()
    print(time.time())

    # agent.play()
    


if __name__ == '__main__':        
    main()
