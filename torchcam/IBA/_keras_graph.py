# From https://github.com/albermax/innvestigate/blob/master/innvestigate/utils/keras/graph.py
# small changes by Leon Sixt
#
#
# COPYRIGHT
#
# The copyright in this software is being made available under the BSD
# License, included below. This software is subject to other contributor rights,
# including patent rights, and no such rights are granted under this license.
#
# All contributions by TU Berlin:
# Copyright (c) 2017-2018 Maximilian Alber, Miriam Haegele, Philipp Seegerer, Kristof T. Schuett
# All rights reserved.
#
# All contributions by Fraunhofer Heinrich Hertz Institute:
# Copyright (c) 2018, Sebastian Lapuschkin
# All rights reserved. Patent pending.
#
# All other contributions:
# Copyright (c) 2018, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
#
# LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import six
import keras
from keras.layers import Activation
# from keras.models import Model
from keras.engine.network import Network
import numpy as np


def to_list(l):
    """ If not list, wraps parameter into a list."""
    if not isinstance(l, list):
        return [l, ]
    else:
        return l


def contains_activation(layer, activation=None):
    """
    Check whether the layer contains an activation function.
    activation is None then we only check if layer can contain an activation.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "activation"):
        if activation is not None:
            return layer.activation == keras.activations.get(activation)
        else:
            return True
    elif isinstance(layer, keras.layers.ReLU):
        if activation is not None:
            return (keras.activations.get("relu") ==
                    keras.activations.get(activation))
        else:
            return True
    elif isinstance(layer, (
            keras.layers.advanced_activations.ELU,
            keras.layers.advanced_activations.LeakyReLU,
            keras.layers.advanced_activations.PReLU,
            keras.layers.advanced_activations.Softmax,
            keras.layers.advanced_activations.ThresholdedReLU)):
        if activation is not None:
            raise Exception("Cannot detect activation type.")
        else:
            return True
    else:
        return False


def is_network(layer):
    """
    Is network in network?
    """
    return isinstance(layer, Network)


def get_symbolic_weight_names(layer, weights=None):
    """Attribute names for weights

    Looks up the attribute names of weight tensors.

    :param layer: Targeted layer.
    :param weights: A list of weight tensors.
    :return: The attribute names of the weights.
    """

    if weights is None:
        weights = layer.weights

    good_guesses = [
        "kernel",
        "bias",
        "gamma",
        "beta",
        "moving_mean",
        "moving_variance",
        "depthwise_kernel",
        "pointwise_kernel"
    ]

    ret = []
    for weight in weights:
        for attr_name in good_guesses+dir(layer):
            if(hasattr(layer, attr_name) and
               id(weight) == id(getattr(layer, attr_name))):
                ret.append(attr_name)
                break
    if len(weights) != len(ret):
        raise Exception("Could not find symoblic weight name(s).")

    return ret


def update_symbolic_weights(layer, weight_mapping):
    """Updates the symbolic tensors of a layer

    Updates the symbolic tensors of a layer by replacing them.

    Note this does not update the loss or anything alike!
    Use with caution!

    :param layer: Targeted layer.
    :param weight_mapping: Dict with attribute name and weight tensors
      as keys and values.
    """

    trainable_weight_ids = [id(x) for x in layer._trainable_weights]
    non_trainable_weight_ids = [id(x) for x in layer._non_trainable_weights]

    for name, weight in six.iteritems(weight_mapping):
        current_weight = getattr(layer, name)
        current_weight_id = id(current_weight)

        if current_weight_id in trainable_weight_ids:
            idx = trainable_weight_ids.index(current_weight_id)
            layer._trainable_weights[idx] = weight
        else:
            idx = non_trainable_weight_ids.index(current_weight_id)
            layer._non_trainable_weights[idx] = weight

        setattr(layer, name, weight)


def get_layer_from_config(old_layer,
                          new_config,
                          weights=None,
                          reuse_symbolic_tensors=True):
    """Creates a new layer from a config

    Creates a new layer given a changed config and weights etc.

    :param old_layer: A layer that shall be used as base.
    :param new_config: The config to create the new layer.
    :param weights: Weights to set in the new layer.
      Options: np tensors, symbolic tensors, or None,
      in which case the weights from old_layers are used.
    :param reuse_symbolic_tensors: If the weights of the
      old_layer are used copy the symbolic ones or copy
      the Numpy weights.
    :return: The new layer instance.
    """
    new_layer = old_layer.__class__.from_config(new_config)

    if weights is None:
        if reuse_symbolic_tensors:
            weights = old_layer.weights
        else:
            weights = old_layer.get_weights()

    if len(weights) > 0:
        input_shapes = old_layer.get_input_shape_at(0)
        # todo: inspect and set initializers to something fast for speedup
        new_layer.build(input_shapes)

        is_np_weight = [isinstance(x, np.ndarray) for x in weights]
        if all(is_np_weight):
            new_layer.set_weights(weights)
        else:
            if any(is_np_weight):
                raise ValueError("Expect either all weights to be "
                                 "np tensors or symbolic tensors.")

            symbolic_names = get_symbolic_weight_names(old_layer)
            update = {name: weight
                      for name, weight in zip(symbolic_names, weights)}
            update_symbolic_weights(new_layer, update)

    return new_layer


def copy_layer_wo_activation(layer,
                             keep_bias=True,
                             name_template=None,
                             weights=None,
                             reuse_symbolic_tensors=True,
                             **kwargs):
    """Copy a Keras layer and remove the activations

    Copies a Keras layer but remove potential activations.

    :param layer: A layer that should be copied.
    :param keep_bias: Keep a potential bias.
    :param weights: Weights to set in the new layer.
      Options: np tensors, symbolic tensors, or None,
      in which case the weights from old_layers are used.
    :param reuse_symbolic_tensors: If the weights of the
      old_layer are used copy the symbolic ones or copy
      the Numpy weights.
    :return: The new layer instance.
    """
    config = layer.get_config()
    if name_template is None:
        config["name"] = None
    else:
        config["name"] = name_template % config["name"]
    if contains_activation(layer):
        config["activation"] = None
    if keep_bias is False and config.get("use_bias", True):
        config["use_bias"] = False
        if weights is None:
            if reuse_symbolic_tensors:
                weights = layer.weights[:-1]
            else:
                weights = layer.get_weights()[:-1]
    return get_layer_from_config(layer, config, weights=weights, **kwargs)


def pre_softmax_tensors(Xs, should_find_softmax=True):
    """Finds the tensors that were preceeding a potential softmax."""
    softmax_found = False

    Xs = to_list(Xs)
    ret = []
    for x in Xs:
        layer, node_index, tensor_index = x._keras_history

        if contains_activation(layer, activation="softmax"):
            softmax_found = True
            if isinstance(layer, Activation):
                ret.append(layer.get_input_at(node_index))
            else:
                layer_wo_act = copy_layer_wo_activation(layer)
                ret.append(layer_wo_act(layer.get_input_at(node_index)))

    if should_find_softmax and not softmax_found:
        raise Exception("No softmax found.")

    return ret
