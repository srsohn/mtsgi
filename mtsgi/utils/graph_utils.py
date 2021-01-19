from typing import Optional, List

import os
import numpy as np
import tensorflow as tf

from acme import specs
from acme.utils import paths

from acme import specs, types


def _sample_int_layer_wise(nbatch, high, low):
  assert(high.ndim == 1 and low.ndim == 1)
  ndim = len(high)
  out_list = []
  for d in range(ndim):
    out_list.append( np.random.randint(low[d], high[d]+1, (nbatch,1 ) ) )
  return np.concatenate(out_list, axis=1)

def get_index_from_pool(pool_ids: np.ndarray, pool_to_index: np.ndarray):
  # (Batched) operation of index = pool_to_index[pool_ids]
  if pool_ids.ndim == 1 and pool_to_index.ndim == 1:
    pool_index = pool_to_index[pool_ids]
  elif pool_ids.ndim == 2 and pool_to_index.ndim == 2:
    assert pool_ids.shape[0] == pool_to_index.shape[0], 'Error: batch dimension is different!'
    pool_index = np.take_along_axis(arr=pool_to_index, indices=pool_ids, axis=1)
  else:
    assert False, 'Error: the shape of "pool_ids" and "pool_to_index" should be both either a) 1-dimensional or b) 2-dimensional!'
  return pool_index

def get_pool_from_index(indices: np.ndarray, index_to_pool: np.ndarray):
  # (Batched) operation of index = index_to_pool[pool_ids]
  if indices.ndim == 1 and index_to_pool.ndim == 1:
    pool_ids = index_to_pool[indices]
  elif indices.ndim == 2 and index_to_pool.ndim == 2:
    assert indices.shape[0] == index_to_pool.shape[0], 'Error: batch dimension is different!'
    pool_ids = np.take_along_axis(arr=index_to_pool, indices=indices, axis=1)
  else:
    assert False, 'Error: the shape of "indices" and "index_to_pool" should be both either a) 1-dimensional or b) 2-dimensional!'
  return pool_ids

def map_index_arr_to_pool_arr(arr_by_index: np.ndarray, pool_to_index: np.ndarray, default_val: int = 0):
  """
    Maps from "array indexed by 'index'" to "array indexed by 'pool-id'".
    E.g., arr_by_index = [1, 2, 3, 4], pool_to_index = [0, -1, -1, 3, -1, 2, 1] ==> arr_by_pool = [1, 0, 0, 4, 0, 3, 2]
    assert no duplication except -1 in pool_to_index
    max(pool_to_index) + 1 == len(arr_by_index) == 13 (for playground)
  """
  assert not isinstance(arr_by_index, tf.Tensor)
  if isinstance(arr_by_index, np.ndarray) and arr_by_index.ndim == 2:
    batch_size = arr_by_index.shape[0]
    if pool_to_index.ndim == 1:
      pool_to_index = np.expand_dims(pool_to_index, 0)
    extended_arr = np.append(arr_by_index, np.full((batch_size, 1), default_val).astype(arr_by_index.dtype), axis=-1)
    arr_by_pool = np.take_along_axis(extended_arr, pool_to_index, axis=-1)
  else:
    extended_arr = np.append(arr_by_index, default_val)
    arr_by_pool = extended_arr[pool_to_index]
  return arr_by_pool

def batched_mapping_expand(arr: np.ndarray, mapping: np.ndarray, default_val: int = 0):
  return map_index_arr_to_pool_arr(arr_by_index=arr, pool_to_index=mapping, default_val=default_val)

def map_pool_arr_to_index_arr(arr_by_pool: np.ndarray, index_to_pool: np.ndarray):
  # assert no duplication except -1 in index_to_pool
  # max(index_to_pool) + 1 == len(arr_by_pool) == 16 (for playground)
  if arr_by_pool.ndim == 2:
    if index_to_pool.ndim == 1:
      index_to_pool = np.expand_dims(index_to_pool, 0)
    arr_by_index = np.take_along_axis(arr_by_pool, index_to_pool, axis=-1)
  elif arr_by_pool.ndim == 1:
    #assert arr_by_index.ndim == 1
    arr_by_index = arr_by_pool[index_to_pool]
  else:
    assert False, f"arr_by_pool.ndim should be either 1 or 2 but got {arr_by_pool.ndim}"
  return arr_by_index

def transform_obs(observation: types.NestedArray, index_to_pool: np.ndarray):
  indexed_obs = {}
  for key, val in observation.items():
    if key in ['mask', 'completion', 'eligibility']:
      indexed_obs[key] = map_pool_arr_to_index_arr(arr_by_pool=val, index_to_pool=index_to_pool)
    else:
      indexed_obs[key] = val
  return indexed_obs

def to_multi_hot(index_tensor, max_dim):
  # number-to-onehot or numbers-to-multihot
  if len(index_tensor.shape)==1:
    out = (np.expand_dims(index_tensor, axis=1) == \
           np.arange(max_dim).reshape(1, max_dim))
  else:
    out = (index_tensor == np.arange(max_dim).reshape(1, max_dim))
  return out

def sample_subtasks(
    rng: np.random.RandomState,
    pool: List[str],
    minimum_size: int,
    maximum_size: Optional[int] = None,
    replace: bool = False
) -> List[str]:
  if maximum_size is not None:
    assert maximum_size <= len(pool), 'Invalid maximum_size.'
  maximum_size = maximum_size or len(pool)
  random_size = rng.randint(minimum_size, maximum_size + 1)
  sampled_subtasks = rng.choice(pool, size=random_size, replace=replace)
  return list(sampled_subtasks)

def add_sampled_nodes(
    graph: 'logic_graph.SubtaskLogicGraph',
    rng: np.random.RandomState,
    pool: List[str],
    minimum_size: int = 1,
    maximum_size: Optional[int] = None
):
  valid_nodes = list(graph.nodes)

  # Sample distractors.
  distractors = sample_subtasks(
      rng=rng,
      pool=pool,
      minimum_size=minimum_size,
      maximum_size=maximum_size
  )

  distractors_added = []
  for distractor in distractors:
    if distractor not in graph:
      distractor_at = np.random.choice(valid_nodes)
      graph[distractor] = distractor_at
      distractors_added.append(distractor)
  return graph, distractors_added


from dataclasses import dataclass

@dataclass
class SubtaskGraph:
  env_id: str
  task_index: int
  num_data: np.ndarray
  numP: np.ndarray
  numA: np.ndarray
  index_to_pool: np.ndarray
  pool_to_index: np.ndarray
  subtask_reward: np.ndarray
  W_a: np.ndarray
  W_o: np.ndarray
  ORmat: np.ndarray
  ANDmat: np.ndarray
  tind_by_layer: list

  def __init__(
      self,
      env_id: Optional[str] = None,
      task_index: Optional[int] = None,
      num_data: Optional[int] = 0,
      numP: Optional[np.ndarray] = None,
      numA: Optional[np.ndarray] = None,
      index_to_pool: Optional[np.ndarray] = None,
      pool_to_index: Optional[np.ndarray] = None,
      subtask_reward: Optional[np.ndarray] = None,
      W_a: Optional[np.ndarray] = None,
      W_o: Optional[np.ndarray] = None,
      ORmat: Optional[np.ndarray] = None,
      ANDmat: Optional[np.ndarray] = None,
      tind_by_layer: Optional[list] = None,
  ):
    self.env_id = env_id
    self.task_index = task_index
    self.num_data = num_data
    self.numP = numP
    self.numA = numA
    self.W_a = W_a
    self.W_o = W_o
    self.index_to_pool = index_to_pool
    self.pool_to_index = pool_to_index
    self.ORmat = ORmat
    self.ANDmat = ANDmat
    self.subtask_reward = subtask_reward
    self.tind_by_layer = tind_by_layer

  def fill_edges(
      self,
      ANDmat: Optional[np.ndarray] = None,
      ORmat: Optional[np.ndarray] = None,
      W_a: Optional[np.ndarray] = None,
      W_o: Optional[np.ndarray] = None,
      tind_by_layer: Optional[list] = None,
  ):
    self.ANDmat = ANDmat
    self.ORmat = ORmat
    self.W_a = W_a
    self.W_o = W_o
    self.tind_by_layer = tind_by_layer
    self.numP, self.numA = [], []
    for tind_list in tind_by_layer:
      self.numP.append(len(tind_list))
    for wa_row in self.W_a:
      self.numA.append(wa_row.shape[0])

class dotdict(dict):
  """dot.notation access to dictionary attributes"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


class GraphVisualizer:
  def __init__(self):
    pass

  def set_num_subtasks(self, num_subtasks: int):
    self._num_subtasks = num_subtasks

  def render_and_save(self, g: 'graphviz.Digraph', path: str):
    g.render(filename=path)
    print('Saved graph @', path)
    return self

  def make_digraph(self) -> 'graphviz.Digraph':
    from graphviz import Digraph
    dot = Digraph(comment='subtask graph', format='pdf')
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(nodesep="0.2", ranksep="0.3")
    dot.node_attr.update(fontsize="14", fontname='Arial')
    return dot

  SUBTASK_NODE_STYLE = dict()
  OPERATOR_NODE_STYLE = dict(shape='rect', style='filled',
                             height="0.15", width="0.15", margin="0.03")

  def visualize_logicgraph(self, g: 'mtsgi.envs.logic_graph.SubtaskLogicGraph'
                           ) -> 'graphviz.Digraph':
    import mtsgi.envs.logic_graph
    LogicOp = mtsgi.envs.logic_graph.LogicOp

    dot = self.make_digraph()
    def _visit_node(node: LogicOp, to: str, has_negation=False):
      # TODO: This access private properties of LogicOp too much.
      # definitely we should move this to logic_graph?
      if node._op_type == LogicOp.TRUE:
        #v_true = f'true_{id(node)}'
        #dot.edge(v_true, to, style='filled')
        pass
      elif node._op_type == LogicOp.FALSE:
        v_false = f'_false_'
        dot.edge(v_false, to, style='filled', shape='rect')
      elif node._op_type == LogicOp.LEAF:
        leaf = node._children[0]
        dot.edge(leaf.name, to, style=has_negation and 'dashed' or '')
      elif node._op_type == LogicOp.NOT:
        op: LogicOp = node._children[0]
        _visit_node(op, to=to, has_negation=not has_negation)
      elif node._op_type == LogicOp.AND:
        v_and = f'and_{to}_{id(node)}'
        dot.node(v_and, "&", **self.OPERATOR_NODE_STYLE)
        dot.edge(v_and, to, style=has_negation and 'dashed' or '')
        for child in node._children:
          _visit_node(child, to=v_and)
        pass
      elif node._op_type == LogicOp.OR:
        v_or = f'or_{to}_{id(node)}'
        dot.node(v_or, "|", **self.OPERATOR_NODE_STYLE)
        dot.edge(v_or, to, style=has_negation and 'dashed' or '')
        for child in node._children:
          _visit_node(child, to=v_or)
      else:
        assert False, str(node._op_type)

    for name, node in g._nodes.items():
      assert name == node.name
      dot.node(node.name)
      _visit_node(node.precondition, to=name)
    return dot

  def visualize(self, cond_kmap_set, subtask_layer,
                subtask_label: List[str]) -> 'graphviz.Digraph':
    """Visualize the subtask graph given its eligibility logic expression.

    Args:
      cond_kmap_set: A sequence of eligibility CNF notations.
        cond_kmap_set[i] = list of clauses, each of which represents
        a vector c where c[j] consists of either {-1, 0, 1}.
        For example, eligibility[i] = c1 OR c2 where
        e.g. c1 = [0, 0, 1, 0, -1]:  (#2) AND (NOT #4)
             c2 = [0, 1, 0, -1]   :  (#1) AND (NOT #3)
    """
    dot = self.make_digraph()

    #cond_kmap_set
    for ind in range(self._num_subtasks):
      if subtask_layer[ind] > -2:
        label = subtask_label[ind]
        dot.node('OR'+str(ind), label, shape='rect', height="0.2", width="0.2", margin="0")
        Kmap_tensor = cond_kmap_set[ind]
        if Kmap_tensor is None:
          continue
        numA, feat_dim = Kmap_tensor.shape
        for aind in range(numA):
          anode_name = 'AND'+str(ind)+'_'+str(aind)
          dot.node(anode_name, "&", shape='rect', style='filled',
                   height="0.15", width="0.15", margin="0.03")
          # OR-AND
          dot.edge(anode_name, 'OR'+str(ind))
          sub_indices = Kmap_tensor[aind, :].nonzero()[0]
          for sub_ind in sub_indices:
            sub_ind = sub_ind.item()
            target = 'OR'+str(sub_ind)

            if Kmap_tensor[aind, sub_ind] > 0: #this looks wrong but it is correct since we used '>' instead of '<='.
              # AND-OR
              dot.edge(target, anode_name)
            else:
              dot.edge(target, anode_name, style="dashed")
    return dot

  def _count_children(self, sub_indices, target_indices):
    count = 0
    for sub_ind in sub_indices:
      sub_ind = sub_ind.item()
      if sub_ind < 39 and sub_ind in target_indices:
        count += 1
    return count


def batch_bin_encode(bin_tensor):
  assert bin_tensor.dtype == np.bool
  feat_dim = bin_tensor.shape[-1]
  if feat_dim > 63:
    dim = bin_tensor.ndim
    bias = 0
    unit = 50
    if dim == 2:
      NB = bin_tensor.shape[0]
      output = [0] * NB
      num_iter = feat_dim//unit + 1
      for i in range(num_iter):
        ed = min(feat_dim, bias + unit)
        out = batch_bin_encode_64(bin_tensor[:, bias:ed])
        out_list = out.tolist()
        output = [output[j] * pow(2, unit) + val for j, val in enumerate(out_list)]
        bias += unit
        if ed==feat_dim:
          break
      return output

    elif dim == 1:
      output = 0
      num_iter = feat_dim//unit + 1
      for i in range(num_iter):
        ed = min(feat_dim, bias + unit)
        out = batch_bin_encode_64(bin_tensor[bias:ed])
        output = output * pow(2, unit) + out
        bias += unit
        if ed==feat_dim:
          break
      return output
    else:
      raise ValueError("dim = %s" % dim)
  else:
    return batch_bin_encode_64(bin_tensor)


def batch_bin_encode_64(bin_tensor):
  # bin_tensor: Nbatch x dim
  assert isinstance(bin_tensor, np.ndarray)
  assert bin_tensor.dtype == np.bool
  return bin_tensor.dot(
      (1 << np.arange(bin_tensor.shape[-1]))
  ).astype(np.int64)
