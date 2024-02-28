import pytest

import numpy as np

import gym_custom
from gym_custom.wrappers import FlattenObservation
from gym_custom import spaces


@pytest.mark.parametrize('env_id', ['Blackjack-v0', 'KellyCoinflip-v0'])
def test_flatten_observation(env_id):
    env = gym_custom.make(env_id)
    wrapped_env = FlattenObservation(env)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()

    if env_id == 'Blackjack-v0':
        space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        wrapped_space = spaces.Box(-np.inf, np.inf,
                                   [32 + 11 + 2], dtype=np.float32)
    elif env_id == 'KellyCoinflip-v0':
        space = spaces.Tuple((
            spaces.Box(0, 250.0, [1], dtype=np.float32),
            spaces.Discrete(300 + 1)))
        wrapped_space = spaces.Box(-np.inf, np.inf,
                                   [1 + (300 + 1)], dtype=np.float32)

    assert space.contains(obs)
    assert wrapped_space.contains(wrapped_obs)
