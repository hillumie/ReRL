from craftax.craftax_env import make_craftax_env_from_name
from envs.craftax_wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
)

def make_env(env_name, num_envs, test_num_envs, optimistic_reset_ratio, use_optimistic_resets=True):
    basic_env = make_craftax_env_from_name(
            env_name, not use_optimistic_resets
        )
    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if use_optimistic_resets:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=num_envs,
            reset_ratio=min(optimistic_reset_ratio, num_envs),
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=test_num_envs,
            reset_ratio=min(optimistic_reset_ratio, test_num_envs),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=num_envs)
        test_env = BatchEnvWrapper(log_env, num_envs=test_num_envs)

    return env, test_env, env_params