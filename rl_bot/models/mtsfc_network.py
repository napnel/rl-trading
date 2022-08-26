import numpy as np
import tree  # pip install dm_tree
from gym.spaces import Box, Discrete, MultiDiscrete
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions

# TODO (sven): add IMPALA-style option.
# from ray.rllib.examples.models.impala_vision_nets import TorchImpalaVisionNet
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.torch_utils import one_hot

torch, nn = try_import_torch()


class MTSFCNetwork(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).

    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.

    The data flow is as follows:

    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and vaulue heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )

        self.processed_obs_space = (
            self.original_space
            if model_config.get("_disable_preprocessor_api")
            else obs_space
        )

        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, self.original_space, action_space, num_outputs, model_config, name
        )

        self.flattened_input_space = flatten_space(self.original_space)

        # Build the CNN(s) given obs_space's image components.
        self.cnns = {}
        self.one_hot = {}
        self.flatten_dims = {}
        self.flatten = {}
        concat_size = 0
        for i, component in enumerate(self.flattened_input_space):
            size = int(np.product(component.shape))
            hidden = 1
            while hidden < size:
                hidden *= 2

            config = {
                # "fcnet_hiddens": model_config["fcnet_hiddens"],
                "fcnet_hiddens": hidden,
                "fcnet_activation": model_config.get("fcnet_activation"),
                "post_fcnet_hiddens": [],
            }
            self.flatten[i] = ModelCatalog.get_model_v2(
                Box(-1.0, 1.0, (size,), np.float32),
                action_space,
                num_outputs=None,
                model_config=config,
                framework="torch",
                name="flatten_{}".format(i),
            )
            self.flatten_dims[i] = size
            concat_size += self.flatten[i].num_outputs

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
            "fcnet_activation": model_config.get("post_fcnet_activation", "relu"),
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"), float("inf"), shape=(concat_size,), dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack",
        )

        # Actions and value heads.
        self._value_out = None

        # Action-distribution head.
        self.logits_layer = SlimFC(
            in_size=self.post_fc_stack.num_outputs,
            out_size=num_outputs,
            activation_fn=None,
            initializer=torch_normc_initializer(0.01),
        )
        # Create the value branch model.
        self.value_layer = SlimFC(
            in_size=self.post_fc_stack.num_outputs,
            out_size=1,
            activation_fn=None,
            initializer=torch_normc_initializer(0.01),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="torch"
            )
        # Push observations through the different components
        # (CNNs, one-hot + FC, etc..).
        outs = []
        for i, component in enumerate(tree.flatten(orig_obs)):
            nn_out, _ = self.flatten[i](
                SampleBatch(
                    {
                        SampleBatch.OBS: torch.reshape(
                            component, [-1, self.flatten_dims[i]]
                        )
                    }
                )
            )
            outs.append(nn_out)

        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack(SampleBatch({SampleBatch.OBS: out}))

        # Logits- and value branches.
        logits, values = self.logits_layer(out), self.value_layer(out)
        self._value_out = torch.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out
