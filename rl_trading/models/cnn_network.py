from typing import Dict, List

import gym
import numpy as np
from ray.rllib.models.torch.misc import (
    SlimConv2d,
    SlimFC,
    normc_initializer,
    same_padding,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class FullyConnectedNetwork(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
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
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation,
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None,
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                    -1
                ]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
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
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
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


class CNN1dNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        nn.Module.__init__(self)
        self._hidden_layers = nn.Sequential()
        self._logits = None
        self._value_branch = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(3)
            value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res
