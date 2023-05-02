import torch.nn as nn
from numpy import kaiser
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import AppendBiasLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            # self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            # self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNetwork(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        *,
        num_channels: list = [128, 32, 8],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.free_log_std = model_config.get("free_log_std")
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2
        num_inputs, num_features = self.obs_space.shape
        # print(num_channels, kernel_size, dropout)

        layers = []
        for i in range(0, len(num_channels)):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.share_network = nn.Sequential(*layers)
        self.policy_network = nn.Sequential(
            # nn.Linear(num_channels[-1] * num_features, num_channels[-1] * num_features),
            # nn.ReLU(),
            nn.Linear(num_channels[-1] * num_features, num_outputs),
        )
        self.value_network = nn.Sequential(
            # nn.Linear(num_channels[-1] * num_features, num_channels[-1] * num_features),
            # nn.ReLU(),
            nn.Linear(num_channels[-1] * num_features, 1),
        )

        if self.free_log_std:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        # print("num_outputs", num_outputs, self.free_log_std)
        self._flat_features = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        _features = self.share_network(obs)
        self._flat_features = _features.reshape(_features.shape[0], -1)
        logits = self.policy_network(self._flat_features)
        # print("logits", logits.shape)
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        # print("logits", logits.shape)

        return logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._flat_features is not None, "must call forward() first"
        value = self.value_network(self._flat_features).squeeze(1)
        return value
