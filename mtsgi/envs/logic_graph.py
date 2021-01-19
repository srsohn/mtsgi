import abc
import copy
import contextlib

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union


class Node(abc.ABC):
  """An abstract base class for vertices in logic computation graph."""

  def __init__(self, name: str):
    self._name = name
    self._reward = 0
    self._stage = None

  @property
  def name(self) -> str:
    return self._name

  @property
  def reward(self) -> float:
    return self._reward

  def set_reward(self, reward: float):
    self._reward = reward

  def __repr__(self):
    return f"{type(self).__name__}[{self._name}]"

  def __hash__(self):
    return hash(self._name)


class LogicOp:
  """A symbolic operation."""

  TRUE = "TRUE"
  FALSE = "FALSE"
  NOT = "NOT"
  AND = "AND"
  OR = "OR"
  LEAF = "_LEAF"

  def __init__(self, op_type, children=[]):
    self._op_type = op_type
    if op_type != LogicOp.LEAF:
      children = [v.as_op() for v in children]
    self._children = children
    assert op_type in (LogicOp.NOT, LogicOp.AND, LogicOp.OR, LogicOp.LEAF, LogicOp.TRUE, LogicOp.FALSE)
    assert isinstance(children, (list, tuple))
    if op_type == LogicOp.NOT and len(children) != 1:
      raise ValueError("NOT node can have only one input.")
    if op_type == LogicOp.LEAF and len(children) != 1:
      raise ValueError("A leaf can have only one input.")

  def as_op(self):
    return self

  def __repr__(self):
    return "LogicOp[" + str(self) + "]"

  def __str__(self):
    if self._op_type == LogicOp.LEAF:
      return str(self._children[0].name)
    else:
      return "%s(%s)" % (self._op_type,
                         ", ".join(str(v) for v in self._children))

  def __and__(self, other) -> 'LogicOp':
    if self._op_type == LogicOp.AND and other._op_type in [LogicOp.AND, LogicOp.LEAF]:
      return LogicOp(LogicOp.AND, list(self._children) + [other])
    else:
      return LogicOp(LogicOp.AND, [self, other])

  def __or__(self, other) -> 'LogicOp':
    if self._op_type == LogicOp.OR and other._op_type in [LogicOp.OR, LogicOp.LEAF]:
      return LogicOp(LogicOp.OR, list(self._children) + [other])
    else:
      return LogicOp(LogicOp.OR, [self, other])

  def __invert__(self) -> 'LogicOp':
    return LogicOp(LogicOp.NOT, [self])

  def evaluate(self, completion) -> bool:
    if self._op_type == LogicOp.TRUE:
      return True
    elif self._op_type == LogicOp.FALSE:
      return False
    elif self._op_type == LogicOp.LEAF:
      node: Node = self._children[0]
      return completion[node.name]
    elif self._op_type == LogicOp.NOT:
      op: LogicOp = self._children[0]
      return (not op.evaluate(completion))
    elif self._op_type == LogicOp.AND:
      return all(op.evaluate(completion) for op in self._children)
    elif self._op_type == LogicOp.OR:
      return any(op.evaluate(completion) for op in self._children)
    else:
      assert False, str(self._op_type)


class SubtaskVertex(Node):

  _op_type = LogicOp.LEAF

  def __init__(self, name: str, *, removable=False):
    super().__init__(name=name)
    self._precondition = LogicOp(LogicOp.TRUE)      # no dependency!

    # Attributes and metadata
    self.removable = removable

  def as_op(self):
    return LogicOp(LogicOp.LEAF, [self])
  @property
  def _children(self):
    return [self]

  @property
  def precondition(self) -> LogicOp:
    return self._precondition

  def __and__(self, other) -> 'LogicOp':
    return self.as_op().__and__(other)
  def __or__(self, other) -> 'LogicOp':
    return self.as_op().__or__(other)
  def __invert__(self) -> 'LogicOp':
    return self.as_op().__invert__()
  def evaluate(self, completion) -> bool:
    return self.as_op().evaluate(completion=completion)




class SubtaskLogicGraph(Node):
  _nodes: Dict[str, Node]   # TODO: Rename as _subtasks.
  _source: Node
  _sink: Node

  def __init__(self, name: str):
    super().__init__(name=name)
    self._nodes = OrderedDict()
    self._source = None   # does not exist yet
    self._sink = None     # does not exist yet
    self._stage = None

  def __repr__(self):
    return f"SubtaskLogicGraph[{self.name}, {len(self.nodes)} nodes]"

  @property
  def nodes(self) -> Sequence[Node]:
    return tuple(self._nodes.values())

  @property
  def stages(self) -> Sequence[str]:
    """Collect all existing stages in this subgraph."""
    stages = []
    for node in [node for node in self.nodes if node._stage]:
      if node._stage not in stages:
        stages.append(node._stage)
    return stages

  def add_node(self, node: Union[str, Node]) -> Node:
    # Add a leaf node.
    if isinstance(node, str):
      node = SubtaskVertex(name=node)
    elif isinstance(node, Node):
      pass
    else:
      raise TypeError("Unknown type: " + str(type(node)))

    if node.name in self._nodes:
      raise ValueError("Already exists: `{}`".format(node.name))

    self._nodes[node.name] = node
    return node

  def __getitem__(self, name) -> SubtaskVertex:
    return self._nodes[name]

  def __setitem__(self, name: str, logic_expression: Node):
    node = self.add_node(name)
    # TODO: Detect Cycles and raise errors.
    node._precondition = logic_expression.as_op()
    # TODO: Support nested stages (it could have multiple values)
    node._stage = self._stage
    return node

  def __contains__(self, name):
    return name in self._nodes

  def compute_eligibility(self, completion: Dict[str, bool]) -> Dict[str, bool]:
    return {
        v.name: v._precondition.evaluate(completion) for v in self._nodes.values()
    }

  def add_reward(self, name: str, reward: float):
    self._nodes[name].set_reward(reward)

  # Utils
  def print_graph(self, subtask_pool_name_to_id: Dict[str, int]):
    print('===Pre-condition===')
    for tind, name in enumerate(self._nodes.keys()):
      precond = self[name]._predicate
      tid = subtask_pool_name_to_id[name]
      print('[%d, %d] %s:' % (tind, tid, name), precond)

  def visualize_graph(self) -> 'graphviz.Digraph':
    from mtsgi.utils import graph_utils
    return graph_utils.GraphVisualizer().visualize_logicgraph(self)

  def connect_nodes(self, sources: Union[str, List[str]], sinks: Union[str, List[str]]):
    if isinstance(sources, str):  # single source node
      predicate = self[sources]
    elif isinstance(sources, list):
      assert len(sources) > 0, 'Sources cannot be empty.'
      predicate = self[sources[0]]
      if len(sources) > 1:
        for elem_name in sources[1:]:
          predicate = predicate & self[elem_name]
    else:
      raise ValueError('Source must be either of type string or list')

    if isinstance(sinks, str):  # single sink node
      self[sinks] = predicate
    elif isinstance(sinks, list):
      for sink in sinks:
        self[sink] = predicate
    else:
      raise ValueError('Sink must be either of type string or list')

  # TODO: delete add_*_to_* methods and use connect_nodes (above).
  def add_many_to_one(self, sources: List[str], sink: str):
    assert len(sources) > 0, 'Sources cannot be empty.'
    predicate = self[sources[0]]
    if len(sources) >= 2:
      for elem_name in sources[1:]:
        predicate = predicate & self[elem_name]
    self[sink] = predicate

  def add_many_to_many(self, sources: List[str], sinks: List[str]):
    assert len(sources) > 0, 'Sources cannot be empty.'
    predicate = self[sources[0]]
    if len(sources) >= 2:
      for elem_name in sources[1:]:
        predicate = predicate & self[elem_name]

    for sink in sinks:
      self[sink] = predicate

  def add_base(self, names: List[str], **kwargs):
    for name in names:
      self.add_node(name, **kwargs)

  def add_one_to_many(self, source: str, sinks: List[str]):
    for elem_name in sinks:
      self[elem_name] = self[source]

  @contextlib.contextmanager
  def stage(self, name):
    try:
      old_stage = self._stage
      self._stage = name
      yield
    finally:
      self._stage = old_stage

  # --- Perturbation operators ---
  # TODO: Write tests.

  def _replace_subtask_preconditions(self,
                                     del_targets: Sequence[SubtaskVertex],
                                     replace_with=None):
    assert all(isinstance(v, SubtaskVertex) for v in del_targets), str(del_targets)

    # Recursive replace
    def _transform(op: LogicOp, under_disjunction=False):
      assert not isinstance(op, Node), f"Must be wrapped with LogicOp. Got: {op}"
      if op._op_type == LogicOp.LEAF:
        if op._children[0] in del_targets:
          if replace_with:
            return replace_with.as_op()
          else:
            return (LogicOp(LogicOp.FALSE) if under_disjunction else
                    _merge_preconditions(del_targets))
        else:
          return op
      else:
        op = copy.copy(op)
        # If an OR node that has 2 or more children, need to replace with
        # constant FALSE. Otherwise, replace with constant TRUE.
        is_disjunction = (op._op_type == LogicOp.OR and len(op._children) > 1)
        op._children = list(set([_transform(x, under_disjunction=is_disjunction)
                                 for x in op._children]))
      return op

    for subtask in self._nodes.values():
      if subtask in del_targets:
        continue
      subtask._precondition = _transform(subtask._precondition)

  def remove_node(self, name, skip_nonexistent=False):
    if isinstance(name, Node): name = name.name
    elif isinstance(name, str): pass
    else: raise TypeError(str(type(name)))

    if name not in self._nodes:
      if skip_nonexistent:
        return
      else:
        raise ValueError(f"Not found: {name}")
    del_target = self[name]

    self._replace_subtask_preconditions(del_targets=[del_target])
    del self._nodes[name]

  def remove_nodes(self, names, **kwargs):
    assert isinstance(names, (list, tuple, set))
    for name in names:
      self.remove_node(name, **kwargs)

  def replace_nodes(self, before, after):
    # TODO: preserve attribute if adding node?
    if isinstance(before, (str, Node)): before = [before]
    if isinstance(after, (str, Node)): after = [after]
    before = [self[node] if isinstance(node, str) else node for node in before]

    after = [self.add_node(node) if isinstance(node, str) else node for node in after]
    if len(after) != 1:
      raise NotImplementedError("Multi-node replace is not supported yet.")
    after[0]._precondition = _merge_preconditions(before)

    self._replace_subtask_preconditions(del_targets=before,
                                        replace_with=after[0])
    for before_node in before:
      del self._nodes[before_node.name]


def _merge_preconditions(subtasks: Sequence[SubtaskVertex]):
  if len(subtasks) == 0:
    return LogicOp(LogicOp.TRUE)
  elif len(subtasks) == 1:
    return subtasks[0].precondition
  else:
    return LogicOp(LogicOp.AND, list(set(s.precondition for s in subtasks)))
