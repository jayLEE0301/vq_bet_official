from .gather_env import GatherEnv
from .ant import AntEnv


class AntGatherEnv(GatherEnv):
    MODEL_CLASS = AntEnv
