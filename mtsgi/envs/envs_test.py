import sys

import acme
import tree
import dm_env
import pytest
import numpy as np

import mtsgi.envs as envs
from mtsgi.envs import wrappers
from mtsgi.envs.base_config import OPTION_NAMES


@pytest.mark.parametrize("env_id, graph_param, action_dim, max_steps", [
    ('playground', 'D1_train', 16, 2),
    ('mining', 'eval', 26, 2),
])
def test_playground_mining(
    env_id: type,
    graph_param: str,
    action_dim: int,
    max_steps: int):
  # Create environment.
  raw_env = envs.MazeOptionEnv(
      game_name=env_id,
      graph_param=graph_param,
      gamma=0.99
  )

  environment = wrappers.MazeWrapper(raw_env)
  environment = wrappers.FewshotWrapper(environment, max_steps=max_steps)
  environment_spec = acme.specs.make_environment_spec(environment)

  observation_spec = environment_spec.observations
  action_spec = environment_spec.actions
  print("observation_spec: ", observation_spec)
  print("action_spec: ", action_spec)

  # Verify observation and action spaces.
  assert action_spec.num_values == action_dim

  assert isinstance(observation_spec, dict)
  assert 'mask' in observation_spec
  assert observation_spec['mask'].shape == (action_dim, )

  # Verify if actual TimeStep values are compliant with the spec.
  ts = environment.reset()
  assert isinstance(ts, dm_env.TimeStep)
  assert ts.first()
  print(ts)

  # TODO: WobConfig and MazeEnv are yet incompatible in terms of API
  # e.g. reset_task, config, etc. Make it unified.


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
