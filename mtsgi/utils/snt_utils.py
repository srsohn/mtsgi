from typing import Optional, Text

from mtsgi.utils import acme_utils

from acme import specs
from acme.tf import networks

import sonnet as snt
import tensorflow as tf



class RecurrentNN(snt.RNNCore):
  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      name: Optional[Text] = None
  ):
    super().__init__(name=name)
    # TODO: make a flags for hidden layer dims.
    self.flat = snt.nets.MLP([64, 64], name="mlp_1")
    self.rnn = snt.DeepRNN([
        snt.nets.MLP([50, 50], activate_final=True, name="mlp_2"),
        snt.GRU(512, name="gru"),
        networks.PolicyValueHead(action_spec.num_values)
    ])

  @tf.function
  def __call__(self, inputs, prev_state):
    _, flat_ob = acme_utils.preprocess_observation(inputs)

    # Process flat observations.
    feat = self.flat(flat_ob)
    feat = tf.nn.relu(feat)
    outputs, new_state = self.rnn(feat, prev_state)
    return outputs, new_state

  def initial_state(self, batch_size):
    return self.rnn.initial_state(batch_size)


class CombinedNN(snt.RNNCore):
  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      name: Optional[Text] = None
  ):
    super().__init__(name=name)

    # Spatial
    self.conv1 = snt.Conv2D(16, 1, 1, data_format="NHWC", name="conv_1")
    self.conv2 = snt.Conv2D(32, 3, 1, data_format="NHWC", name="conv_2")
    self.conv3 = snt.Conv2D(64, 3, 1, data_format="NHWC", name="conv_3")
    self.conv4 = snt.Conv2D(32, 3, 1, data_format="NHWC", name="conv_4")
    self.flatten = snt.Flatten()

    self.fc1 = snt.Linear(256, name="fc_1")

    # Flat
    self.flat = snt.nets.MLP([64, 64], name="mlp_1")
    self.rnn = snt.DeepRNN([
        snt.nets.MLP([50, 50], activate_final=True, name="mlp_2"),
        snt.GRU(512, name="gru"),
       networks.PolicyValueHead(action_spec.num_values)
    ])

  @tf.function
  def __call__(self, inputs, prev_state):
    spatial_ob, flat_ob = acme_utils.preprocess_observation(inputs)

    # TODO: use gpu and switch data_format NHWC --> NCHW
    spatial_ob = tf.transpose(spatial_ob, perm=[0, 2, 3, 1])
    spatial_output = self.conv1(spatial_ob)
    spatial_output = tf.nn.relu(spatial_output)

    spatial_output = self.conv2(spatial_output)
    spatial_output = tf.nn.relu(spatial_output)

    spatial_output = self.conv3(spatial_output)
    spatial_output = tf.nn.relu(spatial_output)

    spatial_output = self.conv4(spatial_output)
    spatial_output = tf.nn.relu(spatial_output)
    spatial_output = self.flatten(spatial_output)

    spatial_output = self.fc1(spatial_output)
    spatial_output = tf.nn.relu(spatial_output)

    # Process flat observations.
    flat_output = self.flat(flat_ob)
    flat_output = tf.nn.relu(flat_output)

    feat = tf.concat([spatial_output, flat_output], axis=-1)
    outputs, new_state = self.rnn(feat, prev_state)
    return outputs, new_state

  def initial_state(self, batch_size):
    return self.rnn.initial_state(batch_size)
