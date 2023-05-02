from typing import Callable

import gym
import numpy as np
import torch.nn as nn
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType


class SimpleFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        initializer: Callable = None,
        activation_fn: Callable = None,
        use_bias: bool = True,
        bias_init: float = 0.0,
        use_batch_norm: bool = False,
        use_dropout: bool = False,
    ):
        super(SimpleFC, self).__init__()
        layers = []
        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer is None:
            initializer = nn.init.xavier_uniform_
        initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_size))

        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())

        if use_dropout:
            layers.append(nn.Dropout(0.2))
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)


class BatchNormModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        use_batch_norm: bool = True,
        use_dropout: bool = True,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens:
            layers.append(
                SimpleFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                    use_batch_norm=use_batch_norm,
                    use_dropout=use_dropout,
                )
            )
            prev_layer_size = size

        self._logits = SimpleFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SimpleFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0),
                        use_batch_norm=use_batch_norm,
                        use_dropout=use_dropout,
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SimpleFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))

        self._hidden_layers.train(mode=is_training)
        if self._value_branch_separate:
            self._value_branch_separate.train(mode=is_training)

        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)
