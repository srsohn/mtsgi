# Expose base/sge envrionment.
from mtsgi.envs.base import BaseWoBEnv
from mtsgi.envs.sge.mazeenv import MazeEnv, MazeOptionEnv

from mtsgi.envs import logic_graph

# Expose environment configurations.
from mtsgi.envs.toywob.toywob_config import ToyWoB


def get_subtask_label(env_id):
  if env_id == 'toywob':
    from mtsgi.envs.toywob.toywob_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'playground':
    # TODO: maybe use subtask name
    return [str(num) for num in range(13)]
  else:
    return ValueError(env_id)
