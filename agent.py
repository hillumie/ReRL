import os
import time
from copy import copy

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen  as nn
from flax.training.train_state import TrainState
from flax import struct
import chex
import optax

import wandb

from functools import partial
from typing import Any
from omegaconf import OmegaConf

from utils.common_utils import init_log, save_state
from utils.cfgs import print_cfgs 



class Agent():
    def __init__(self, config) -> None:
        

        self.config = {**config, **config["alg"]}
        print_cfgs(self.config)

        self.alg_name = self.config.get("ALG_NAME", "pqn")
        self.env_name = self.config["ENV_NAME"]

        init_log(self.config)

        self.pure_jax = True

        start_time = time.time()
        if self.pure_jax:
            if self.config["ALG_NAME"] == "pqn_atari":
                from agents.pqn_atari_agent import make_train
                self.agent = jax.jit(make_train(self.config))
            elif self.config["ALG_NAME"] == "pqn_gymnax":
                from agents.pqn_gymnax_agent import make_train
                self.agent = jax.jit(make_train(self.config))
            elif self.config["ALG_NAME"] == "pqn_craftax":
                from agents.pqn_craftax_agent import make_train
                self.agent = jax.jit(make_train(self.config))
            elif self.config["ALG_NAME"] == "pqn_minatar":
                from agents.pqn_minatar_agent import make_train
                self.agent = jax.jit(make_train(self.config))
            elif self.config["ALG_NAME"] == "pqn_rnd_atari":
                from agents.pqn_rnd_atari_agent import make_train
                self.agent = jax.jit(make_train(self.config))
            else:
                raise NotImplemented(f"{self.config['ALG_NAME']} is not yet implemented")
        else:
            rng = jax.random.PRNGKey(self.config["SEED"])
            if self.config["ALG_NAME"] == "pqn_gymnax":
                from agents.impure_pqn_gymnax_agent import PQNGymnaxAgent
                self.agent = PQNGymnaxAgent(self.config, rng)
            elif self.config["ALG_NAME"] == "ppo_continuous":
                from agents.impure_ppo_continuous_agent import PPOContinuousAgent
                self.agent = PPOContinuousAgent(self.config, rng)
            elif self.config["ALG_NAME"] == "ppo_isaac":
                from agents.impure_ppo_isaaclab_agent import PPOIsaacAgent
                self.agent = PPOIsaacAgent(self.config, rng)
            elif self.config["ALG_NAME"] == "ppo_mshab":
                from agents.impure_ppo_mshab_agent import PPOMSHABAgent
                self.agent = PPOMSHABAgent(self.config, rng)
            else:
                raise NotImplemented(f"{self.config['ALG_NAME']} is not yet implemented")
        
        print(f"Take {time.time() - start_time:.2f} seconds to compile")
    
    def run(self):
        
        t0 = time.time()
        if self.config["NUM_SEEDS"] > 1:
            raise NotImplementedError("Vmapped seeds not supported yet.")
        else:
            if self.pure_jax:
                rng = jax.random.PRNGKey(self.config["SEED"])
                outs = self.agent(rng)
            else:
                outs = self.agent.run()
        print(f"Took {time.time()-t0} seconds to complete.")

        # save hyperparams
        if self.config.get("SAVE_STATE", True):
            save_dir = os.path.join(self.config["SAVE_PATH"], self.env_name)
            os.makedirs(save_dir, exist_ok=True)
            OmegaConf.save(
                self.config,
                os.path.join(
                    save_dir, f'{self.alg_name}_{self.env_name}_seed{self.config["SEED"]}_config.yaml'
                ),
            )

            # # assumes not vmpapped seeds
            # save_state(outs, iteration=outs["metrics"]["general"]["iterations"][-1])
        

    def tune(self, default_config):
        """Hyperparameter sweep with wandb."""

        default_config = {
            **default_config,
            **default_config["alg"],
        }  # merge the alg config with the main config

        def wrapped_make_train():
            wandb.init(project=default_config["PROJECT"])

            # update the default params
            config = copy.deepcopy(default_config)
            for k, v in dict(wandb.config).items():
                config["alg"][k] = v

            print("running experiment with params:", config)

            rng = jax.random.PRNGKey(config["SEED"])

            if config["NUM_SEEDS"] > 1:
                raise NotImplementedError("Vmapped seeds not supported yet.")
            else:
                outs = self.agent(rng)

        sweep_config = {
            "name": f"pqn_atari_{default_config['ENV_NAME']}",
            "method": "bayes",
            "metric": {
                "name": "test_returns",
                "goal": "maximize",
            },
            "parameters": {
                "LR": {"values": [0.0005, 0.0001, 0.00005]},
                "LAMBDA": {"values": [0.3, 0.6, 0.9]},
            },
        }

        wandb.login()
        sweep_id = wandb.sweep(
            sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
        )
        wandb.agent(sweep_id, wrapped_make_train, count=1000)