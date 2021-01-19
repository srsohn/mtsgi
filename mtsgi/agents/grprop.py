"""Baselines Actor implementations."""

from typing import Dict, List
import numpy as np

import dm_env
import acme
from acme import specs, types

from mtsgi.agents import base
from mtsgi.graph.grprop import GRProp
from mtsgi.utils import graph_utils
from mtsgi.utils.graph_utils import SubtaskGraph


class GRPropActor(base.BaseActor):
  """An Actor that chooses random action."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      temp: float = None,
      w_a: float = None,
      beta_a: float = None,
      ep_or: float = None,
      temp_or: float = None,
  ):
    self._observation_spec: types.NestedArray = environment_spec.observations
    self._action_spec: specs.DiscreteArray = environment_spec.actions
    self._grprop = GRProp(
        environment_spec=environment_spec,
        temp=temp,
        w_a=w_a, beta_a=beta_a,
        ep_or=ep_or, temp_or=temp_or
    )
    self._index_to_pool = None
    self._pool_to_index = None
    self._num_data = None

  @property
  def is_ready(self):
    return self._grprop.is_ready

  def observe_task(self, tasks: List[SubtaskGraph]):
    self._grprop.init_graph(graphs=tasks)
    self._index_to_pool = self._grprop._index_to_pool
    self._pool_to_index = np.stack([task.pool_to_index for task in tasks])
    self._num_data = np.stack([task.num_data for task in tasks])

  @property
  def confidence_score(self):
    return self._num_data

  def observe_first(self, timestep: dm_env.TimeStep):
    pass  # Do nothing.

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    pass  # Do nothing.

  def update(self):
    pass  # Do nothing.

  def _get_raw_logits_indexed_debug(self, indexed_obs: Dict[str, np.ndarray]) -> np.ndarray:
    # Run GRProp for test phase.
    logits_indexed = self._grprop.get_raw_logits(
        observation=indexed_obs
    )
    return logits_indexed

  def get_availability(self, observation: Dict[str, np.ndarray]) -> bool:
    if not self.is_ready:
      return np.full(shape=(observation['mask'].shape[0]), fill_value=False, dtype=np.bool)
    indexed_obs = graph_utils.transform_obs(
        observation=observation,
        index_to_pool=self._index_to_pool,
    )
    mask = indexed_obs['mask']
    eligibility = indexed_obs['eligibility']
    masked_eligibility = np.multiply(mask, eligibility)
    return masked_eligibility.sum(-1) > 0.5

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    indexed_obs = graph_utils.transform_obs(
        observation=observation,
        index_to_pool=self._index_to_pool,
    )

    # Run GRProp for test phase.
    logits_indexed = self._grprop.get_raw_logits(
        observation=indexed_obs
    )

    # index_to_pool
    # TODO: Subtask pools (e.g. 89) might differ from options (e.g. 91),
    # so this is technically wrong.
    logits = graph_utils.map_index_arr_to_pool_arr(
        logits_indexed,
        pool_to_index=self._pool_to_index,
        default_val=0.
    )
    assert self._action_spec.num_values == logits.shape[1], str(logits.shape)
    return logits
