from .ant_gather_env import AntGatherEnv


def create_gather_env(env_name=None, seed=0):
    if env_name.startswith("Ant"):
        cls = AntGatherEnv
        env_name = env_name[3:]
    else:
        assert False, "unknown env %s" % env_name

    gym_mujoco_kwargs = {"seed": seed}
    gym_env = cls(**gym_mujoco_kwargs)
    gym_env.reset()
    return gym_env
