"""Baselines Actor implementations."""

from typing import Dict, List, Sequence
from collections import defaultdict

import abc
import numpy as np

import dm_env
import acme
from acme import specs, types
from acme.tf import utils as tf2_utils

from mtsgi.utils import graph_utils, tf_utils
from mtsgi.utils.graph_utils import SubtaskGraph


class BaseActor(acme.Actor):
  """Actor Base class
  """
  __metaclass__ = abc.ABC

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      verbose_level: int = 0,
  ):
    self._observation_spec: types.NestedArray = environment_spec.observations
    self._action_spec: specs.DiscreteArray = environment_spec.actions
    self._verbose_level = verbose_level

  def observe_task(self, tasks: List[SubtaskGraph]):
    pass # Do nothing upon starting a new task

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

  def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    masked_eligibility = np.multiply(observation['mask'], observation['eligibility'])

    # Choose from stochastic policy
    termination = observation['termination']
    logits = self._get_raw_logits(observation)

    # When the environment is done, replace with uniform policy. This action
    # will be ignored by the environment
    if np.any(termination):
      masked_eligibility[termination, :] = 1
      logits[termination, :] = 1.

    # Masking
    probs = self._mask_softmax(logits, masked_eligibility)
    actions = tf_utils.categorical_sampling(probs=probs).numpy()

    if self._verbose_level > 0:
      print(f'actions (pool)={actions[0].item()}')
      print('probs (pool)=', probs[0])
    if self._verbose_level > 1:
      print('masked_eligibility (pool)= ', masked_eligibility[0])

    return actions

  @abc.abstractmethod
  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    raise NotImplementedError

  def _mask_logits(self, logits: np.array, masked_eligibility: np.array) -> np.ndarray:
    # Numerically stable masking
    # Ex) If Mask = [1, 0, 1]. logits=[1, 0, -5]->[6, 5, 0]->[6, -infty, 0]->[0, -infty, -6]
    # Ex) If Mask = [1, 1, 0]. logits=[1, 0, -5]->[6, 5, 0]->[6, 5, -infty]->[0, -1, -infty]
    out_logits = logits - logits.min(axis=-1, keepdims=True)
    out_logits[masked_eligibility == 0] = -np.infty
    out_logits -= out_logits.max(axis=-1, keepdims=True)
    assert not np.any(np.isnan(out_logits)), 'Error! Nan in logits'
    return out_logits

  def _mask_softmax(self, logits: np.array, masked_eligibility: np.ndarray) -> np.ndarray:
    # Numerically stable masked-softmax
    masked_logits = self._mask_logits(logits, masked_eligibility)
    exps = np.exp(masked_logits)
    probs_masked = exps / exps.sum(axis=-1, keepdims=True)

    # Make sure by masking probs once more
    probs_masked = probs_masked * masked_eligibility
    probs_masked = probs_masked / probs_masked.sum(axis=-1, keepdims=True)
    return probs_masked


class FixedActor(BaseActor):
  """An Actor that chooses a predefined set of actions."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      option_sequence: Sequence[np.ndarray],
      verbose_level: int = 0,
  ):
    super().__init__(environment_spec=environment_spec, verbose_level=verbose_level)
    self.step_count = 0
    self.option_seq = option_sequence

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    action = self.option_seq[self.step_count]
    self.step_count += 1
    logits = np.zeros_like(observation['mask'])
    logits[:, action] = 1.
    return logits


class RandomActor(BaseActor):
  """An Actor that chooses random action."""

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['mask'].ndim == 2 and observation['eligibility'].ndim == 2
    logits = np.ones_like(observation['mask'])
    return logits


class UCBActor(BaseActor):
  """An Actor that chooses an action using UCB."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      temperature: float,
      verbose_level: int = 0,
  ):
    super().__init__(environment_spec=environment_spec, verbose_level=verbose_level)
    self._temperature = temperature

  def observe_task(self, tasks: List[SubtaskGraph]):
    num_options = tasks[0].pool_to_index.shape[0]
    self._avg_rewards = np.zeros((len(tasks), num_options))
    self._counts = np.ones((len(tasks), num_options))
    self._total_counts = np.full(shape=(len(tasks)), fill_value=num_options)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    action = np.expand_dims(action, axis=-1)
    next_rewards = np.expand_dims(next_timestep.reward, axis=-1)
    is_first = np.expand_dims(next_timestep.first(), axis=-1)  # for mask
    avg_rewards = np.take_along_axis(self._avg_rewards, action, axis=-1)
    counts = np.take_along_axis(self._counts, action, axis=-1)

    # Compute & update avg rewards.
    update_values = 1 / counts * (next_rewards - avg_rewards)
    next_avg_rewards = avg_rewards + np.where(is_first, 0, update_values)  # skip first timestep.
    np.put_along_axis(self._avg_rewards, action, values=next_avg_rewards, axis=-1)

    # Update counts.
    np.put_along_axis(self._counts, action, values=counts + (1 - is_first), axis=-1)
    self._total_counts += (1 - is_first).squeeze()

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['completion'].shape[1] == observation['mask'].shape[1]
    logits = np.zeros_like(observation['mask'])

    masked_eligibility = observation['mask'] * observation['eligibility']
    for i, masked_elig in enumerate(masked_eligibility):
      if not observation['termination'][i]:
        options = masked_elig.nonzero()[0]
        rewards = self._avg_rewards[i][options]
        counts = self._counts[i][options]
        utility = rewards + np.sqrt(2 * np.log(self._total_counts[i]) / counts)
        logits[i][options] = utility / self._temperature

    return logits


class MTUCBActor(UCBActor):
  """An Multi-Task UCB actor."""

  def observe_task(self, tasks: List[SubtaskGraph]):
    pool_to_index = np.stack([g.pool_to_index for g in tasks], axis=0)
    prior_rewards = np.stack([g.subtask_reward for g in tasks], axis=0)
    prior_counts = np.stack([g.reward_count for g in tasks], axis=0)

    # Use prior to initialize the avg rewards.
    self._avg_rewards = graph_utils.map_index_arr_to_pool_arr(
        arr_by_index=prior_rewards,
        pool_to_index=pool_to_index
    )

    # Initialize counts.
    self._counts = graph_utils.map_index_arr_to_pool_arr(
        arr_by_index=prior_counts,
        pool_to_index=pool_to_index
    )
    self._counts += 1  # avoid nan
    self._total_counts = np.sum(self._counts, axis=-1)


class CountBasedActor(BaseActor):
  """An Actor that chooses an action using simple count-based strategy."""

  def observe_task(self, tasks: List[SubtaskGraph]):
    self._count_tables = [defaultdict(lambda: 0)] * len(tasks)

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['completion'].shape[1] == observation['mask'].shape[1]
    logits = np.zeros_like(observation['mask'])
    completions = observation['completion'].copy().astype(np.int8)

    masked_eligibility = observation['mask'] * observation['eligibility']
    for i, (comp, masked_elig) in enumerate(zip(completions, masked_eligibility)):
      if not observation['termination'][i]:
        options = masked_elig.nonzero()[0]
        counts = []
        for option in options:
          comp[option] = 1  # predicted completion.
          counts.append(self._count_tables[i][tuple(comp)])

        min_idx = np.argmin(counts + np.random.random(len(counts)))  # random tie break.
        logits[i][options[min_idx]] = 1000.0  # XXX assign high probability
    return logits


class GreedyActor(BaseActor):
  """An Actor that chooses random action."""

  def observe_task(self, tasks: List[SubtaskGraph]):
    # TODO: currently this only works for playground/mining.
    subtask_rewards_by_id = []
    for task in tasks:
      reward_vec = graph_utils.map_index_arr_to_pool_arr(
          arr_by_index=task.subtask_reward,
          pool_to_index=task.pool_to_index
      )
      reward_vec = reward_vec - reward_vec.min() + 1.0
      subtask_rewards_by_id.append(reward_vec)
    self._subtask_reward = np.stack(subtask_rewards_by_id)

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    # Choose random actions.
    assert  observation['mask'].ndim == 2 and observation['eligibility'].ndim == 2 and self._subtask_reward.ndim == 2
    logits = self._subtask_reward
    return logits
