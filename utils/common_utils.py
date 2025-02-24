import jax.experimental
import numpy as np
import jax
import jax.numpy as jnp
import csv
import os

import logging
import wandb
import time 
import shutil
import psutil
from typing import List, Tuple


import orbax.checkpoint
from flax.training import orbax_utils

import rich
from rich.console import Console
from rich.text import Text
from rich.table import Table
import datetime


# This will contain all the global variables that are not JAX entities
def init_log(config):
    global global_config
    global_config = config

    global save_path
    if config["SAVE_PATH"] != "None":
        save_path = config["SAVE_PATH"]
    else:
        save_path = os.path.join("logs", config["ENV_NAME"], config["ALG_NAME"], f"seed-{config['SEED']}")
    os.makedirs(save_path, exist_ok=True)
    if not config["WANDB_LOG_ALL_SEEDS"] and config["WANDB_MODE"] != "disabled":
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=[
                config["ALG_NAME"].upper(),
                config["ENV_NAME"].upper(),
                f"jax_{jax.__version__}",
            ],
            name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
            config=config,
            mode=config["WANDB_MODE"],
        )

    if config.get("SAVE_STATE", True):
        save_state_path = os.path.abspath(save_path + f"/models")
        os.makedirs(os.path.dirname(save_state_path), exist_ok=True)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        global checkpoint_manager
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            save_state_path, orbax_checkpointer)



CONSOLE_FORMAT = [
	("iterations", "I", "int"),
	("episodes", "E", "int"),
	("env_frames", "F", "int"),
	("returned_episode_returns", "R", "float"),
	("episode_success", "S", "float"),
	("uptime", "T", "time"),
]

def _format(value, ty):
    if ty == "int":
        return f"{int(value):,}"
    elif ty == "float":
        return f"{value:.01f}"
    elif ty == "time":
        value = str(datetime.timedelta(seconds=int(value)))
        return f"{value}"
    else:
        raise f"invalid log format type: {ty}"
		

def casual_print(d):
    console = Console()
    texts = Text()
    for k, disp_k, ty in CONSOLE_FORMAT:
        if k in d:
            d_v = _format(d[k], ty)
            texts.append(f" | {disp_k}: ", 'orange1')
            texts.append(d_v, 'medium_spring_green')
    console.print(texts)


# metrics: steps, episode, generation, rewards, time, loss (policy, value, entropy, KL div), ep length,
# residual variance ((y - y_hat)/var of y), 
# terminal correlation (corr of final reward and value at ter step), penultimate terminal correlation
# value target distribution (histogram, min/max/min/std), reward distribution, value distribution, advantage distribution, ep length distribution
# steps statistic (abs max, mean square value)
# sample staleness (asynchronous on-policy)
# gradient noise (choose batch size) 


metric_headers = ["iterations", "env_steps", "env_frames", "clock_time", "episodes", "grad_steps", 
                    "total_rewards", "episode_length", 
                    "policy_kl_div", "policy_entropy", "residual_var", "terminal_var",
                    "policy_loss", "v_value_loss", "q_value_loss", "penultimate_terminal_corr", 
                    "target_v_values", "rewards", "v_values", "q_values", "misc",
                    ]

init_metrics = dict.fromkeys(metric_headers)
for k in init_metrics.keys():
    init_metrics[k] = None

def plot_results(metrics, config):
    raise NotImplemented


def log_wandb(metrics, original_seed):
    metrics.update(
        {
            f"rng{int(original_seed)}/{k}": v
            for k, v in metrics.items()
        }
    )
    wandb.log(metrics, step=metrics["iterations"])

def log_csv(metrics, original_seed):
    if metrics["iterations"] == 1:
        with open(save_path + "/csv_output.csv", "w") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writeheader()
    else:
        with open(save_path + "/csv_output.csv", "a") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writerow(metrics)

try: # Some modules (envpool) use old CUDA which does not match with torch
    import torch
    from torch.utils.tensorboard import SummaryWriter

    def log_tensorboard(metrics, original_seed):
        writer = SummaryWriter(save_path)
        for k, v in metrics:
            writer.add_scalar(k, v, metrics["iterations"])
except:
    print("ignore tensorboard since torch is not supported")


def log_callback(metrics, global_metrics, original_seed, config):
    general_ = metrics["general"]
    losses_ = metrics["losses"]
    time_ = metrics["time"]
    eval_ = global_metrics["eval"]

    if not config["WANDB_LOG_ALL_SEEDS"] and config["WANDB_MODE"] != "disabled":
        wandb_metrics = dict()
        wandb_metrics.update(general_)
        wandb_metrics.update(losses_)
        wandb_metrics.update(time_)
        wandb_metrics.update(eval_)
        jax.debug.callback(log_wandb, wandb_metrics, original_seed)

    if config.get("CSV_LOG", True):
        csv_metrics = dict()
        csv_metrics.update(general_)
        csv_metrics.update(losses_)
        csv_metrics.update(time_)
        csv_metrics.update(eval_)
        jax.debug.callback(log_csv, csv_metrics, original_seed)

    try:
        if config.get("TENSORBOARD_LOG", True):
            jax.debug.callback(log_tensorboard, metrics, original_seed)
    except:
        print("ignore tensorboard since torch is not supported")

    if config.get("PRINT", True):
        print_callback(metrics, global_metrics, original_seed, config)
        
def print_callback(metrics, global_metrics, original_seed, config):
    all_metrics = dict()
    time_ = metrics["time"]
    losses_ = metrics["losses"]
    all_metrics.update(metrics["general"])
    all_metrics.update(metrics["losses"])
    all_metrics.update(global_metrics)
    all_metrics.update(metrics["misc"])
    other_metrics = metrics["general"]
    other_metrics.update(global_metrics["eval"])
    SPS = (all_metrics["env_frames"] - all_metrics["prev_steps"]) / (time_["uptime"] - global_metrics["uptime"])
    all_metrics.update({"SPS": SPS,
                    "remaining": (config["TOTAL_TIMESTEPS"] * config["ENV_KWARGS"]["frame_skip"] - all_metrics["env_frames"]) / SPS,
    })

    all_metrics["misc_time"] = all_metrics["uptime"] - all_metrics["env_time"] - all_metrics["train_time"] - all_metrics["eval_time"]

    if config.get("DASHBOARD", False):
        jax.debug.callback(print_dashboard, all_metrics, losses_, all_metrics["env_frames"], all_metrics["iterations"], other_metrics, True)
    else:
        jax.debug.callback(casual_print, all_metrics)


def save_state(ckpt, iteration):
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(iteration, ckpt, save_kwargs={'save_args': save_args})



def load_state(iteration=None, latest=False):
    if latest:  # ignore inputted iteration and get latest iteration
        iteration = checkpoint_manager.latest_step()
    
    if iteration is None:
        raise ValueError("Either iteration is not None or lastest is True")
    
    restored = checkpoint_manager.restore(iteration)

    return restored


# Standard JAX only supports 32 bits while time.time() returns a float64 time value precising up to nanosec
# When convert this to JAX type, the time value loses all decimal
# Use time.time_ns() to get full time (up to nanosec) in int32 so that it does not lose precision 
# Split this time into integer and decimal parts
def get_p_time():
    t = time.time_ns()
    t_dec = t%(10**9)
    t_int = (t//(10**9))%(10**9)
    return [np.int32(t_int), np.int32(t_dec)] # Split integer and decimal apart that int32 can handle
    
def get_time():
    return jax.experimental.io_callback(lambda: jnp.array(get_p_time()), jax.eval_shape(lambda: jnp.array([jnp.int32(0), jnp.int32(0)])))

# After subtracting, integer part gets small, the decimal part is regularized to float32 from dividing to 10^9
# Add these 2 parts to have the real time with highest precision possible 
def get_time_diff(time1, time2):
    diff = time1 - time2
    real_time = diff[0] + diff[1]/(10**9)
    return real_time





# Rich printing originates from "https://github.com/PufferAI/PufferLib/blob/1.0/clean_pufferl.py"

ROUND_OPEN = rich.box.Box(
    "╭──╮\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "╰──╯\n"
)

c1 = '[bright_cyan]'
c2 = '[white]'
c3 = '[cyan]'
b1 = '[bright_cyan]'
b2 = '[bright_white]'

def abbreviate(num):
    if num < 1e3:
        return f'{b2}{num:.0f}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}k'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}m'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}b'
    else:
        return f'{b2}{num/1e12:.1f}{c2}t'

def duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100*time/uptime - 1e-5)
    return f'{c1}{name}', duration(time), f'{b2}{percent:2d}%'


# TODO: Add env name to print_dashboard
def print_dashboard(metrics, losses, global_step, epoch,
        stats, clear=False, max_stats=[0]):

    try:
        import torch
        gpu_util = torch.cuda.utilization()
        free, total = torch.cuda.mem_get_info()
        gpu_mem = 100*free/total
    except:
        gpu_util = 0
        gpu_mem = 0

    mem = psutil.virtual_memory()
    metrics.update({"cpu_util": 100*psutil.cpu_percent(),
                "cpu_mem": 100*mem.active/mem.total,
                "gpu_util": gpu_util,
                "gpu_mem": gpu_mem,
    })
    
    console = Console()
    if clear:
        console.clear()

    dashboard = Table(box=ROUND_OPEN, expand=True,
        show_header=False, border_style='bright_cyan')

    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)
    cpu_percent = jnp.mean(metrics["cpu_util"])
    dram_percent = jnp.mean(metrics["cpu_mem"])
    gpu_percent = jnp.mean(metrics["gpu_util"])
    vram_percent = jnp.mean(metrics["gpu_mem"])
    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=13)
    table.add_column(justify="right", width=13)
    table.add_row(
        f'{c1}Dashboard',
        f'{c1}CPU: {c3}{cpu_percent:.1f}%',
        f'{c1}GPU: {c3}{gpu_percent:.1f}%',
        f'{c1}DRAM: {c3}{dram_percent:.1f}%',
        f'{c1}VRAM: {c3}{vram_percent:.1f}%',
    )
        
    s = Table(box=None, expand=True)
    s.add_column(f"{c1}Summary", justify='left', vertical='top', width=16)
    s.add_column(f"{c1}Value", justify='right', vertical='top', width=8)
    s.add_row(f'{c2}Environment', f'{b2}{global_config["ENV_NAME"]}')
    s.add_row(f'{c2}Agent Steps', abbreviate(global_step))
    s.add_row(f'{c2}SPS', abbreviate(metrics["SPS"]))
    s.add_row(f'{c2}Epoch', abbreviate(epoch))
    s.add_row(f'{c2}Uptime', duration(metrics["uptime"]))
    s.add_row(f'{c2}Remaining', duration(metrics["remaining"]))

    p = Table(box=None, expand=True, show_header=False)
    p.add_column(f"{c1}Performance", justify="left", width=10)
    p.add_column(f"{c1}Time", justify="right", width=8)
    p.add_column(f"{c1}%", justify="right", width=4)
    p.add_row(*fmt_perf('Evaluate', metrics["eval_time"], metrics["uptime"]))
    p.add_row(*fmt_perf('  Env', metrics["env_time"], metrics["uptime"]))
    p.add_row(*fmt_perf('Train', metrics["train_time"], metrics["uptime"]))
    p.add_row(*fmt_perf('  Misc', metrics["misc_time"], metrics["uptime"]))

    l = Table(box=None, expand=True, )
    l.add_column(f'{c1}Losses', justify="left", width=16)
    l.add_column(f'{c1}Value', justify="right", width=8)
    for metric, value in losses.items():
        l.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(s, p, l)
    dashboard.add_row(monitor)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    left = Table(box=None, expand=True)
    right = Table(box=None, expand=True)
    table.add_row(left, right)
    left.add_column(f"{c1}User Stats", justify="left", width=20)
    left.add_column(f"{c1}Value", justify="right", width=10)
    right.add_column(f"{c1}User Stats", justify="left", width=20)
    right.add_column(f"{c1}Value", justify="right", width=10)
    i = 0
    for metric, value in stats.items():
        try: # Discard non-numeric values
            int(value)
        except:
            continue

        u = left if i % 2 == 0 else right
        u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
        i += 1

    for i in range(max_stats[0] - i):
        u = left if i % 2 == 0 else right
        u.add_row('', '')

    max_stats[0] = max(max_stats[0], i)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    table.add_row(f' {c1}Message: {c2}Shout out to :blowfish: {b2}Pufferlib {c2}for this cool dashboard')

    with console.capture() as capture:
        console.print(dashboard)

    print('\033[0;0H' + capture.get())