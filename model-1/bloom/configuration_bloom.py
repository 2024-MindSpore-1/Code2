# coding=utf-8
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Bloom configuration"""

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)

BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bigscience/bloom": "https://hf-mirror.com/bigscience/bloom/resolve/main/config.json",
    "bigscience/bloom-560m": "https://hf-mirror.com/bigscience/bloom-560m/blob/main/config.json",
    "bigscience/bloom-1b1": "https://hf-mirror.com/bigscience/bloom-1b1/blob/main/config.json",
    "bigscience/bloom-1b7": "https://hf-mirror.com/bigscience/bloom-1b7/blob/main/config.json",
    "bigscience/bloom-3b": "https://hf-mirror.com/bigscience/bloom-3b/blob/main/config.json",
    "bigscience/bloom-7b1": "https://hf-mirror.com/bigscience/bloom-7b1/blob/main/config.json",
}


class BloomConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BloomModel`]. It is used to instantiate a Bloom
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Bloom architecture
    [bigscience/bloom](https://hf-mirror.com/bigscience/bloom).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 250880):
            Vocabulary size of the Bloom model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`BloomModel`]. Check [this
            discussion](https://hf-mirror.com/bigscience/bloom/discussions/120#633d28389addb8530b406c2a) on how the
            `vocab_size` has been defined.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to [this
            document](https://hf-mirror.com/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). Note also that this is enabled only when
            `slow_but_exact=True`.
        slow_but_exact (`bool`, *optional*, defaults to `False`):
            Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
            merging the TP rank tensors, due to slicing operations the results may be slightly different between the
            model trained on Megatron and our model. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). A solution to obtain more accurate results is to
            enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
            resolved in the future once the main model has been fine-tuned with TP_rank=1.

    Example:
        ```python
        >>> from transformers import BloomConfig, BloomModel
        ...
        >>> # Initializing a Bloom configuration
        >>> configuration = BloomConfig()
        ...
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = BloomModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "bloom"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pretraining_tp=1,  # TP rank used when training with megatron
        slow_but_exact=False,
        **kwargs,
    ):
        """
        Initializes a new instance of the BloomConfig class.
        
        Args:
            self: The object itself.
            vocab_size (int): The size of the vocabulary. Default is 250880.
            hidden_size (int): The size of the hidden layer. Default is 64.
            n_layer (int): The number of layers. Default is 2.
            n_head (int): The number of attention heads. Default is 8.
            layer_norm_epsilon (float): The epsilon value for layer normalization. Default is 1e-05.
            initializer_range (float): The range for the initializer. Default is 0.02.
            use_cache (bool): Determines if caching is used. Default is True.
            bos_token_id (int): The ID of the beginning-of-sentence token. Default is 1.
            eos_token_id (int): The ID of the end-of-sentence token. Default is 2.
            apply_residual_connection_post_layernorm (bool): Determines if residual connection is applied after layer normalization. Default is False.
            hidden_dropout (float): The dropout rate for hidden layers. Default is 0.0.
            attention_dropout (float): The dropout rate for attention layers. Default is 0.0.
            pretraining_tp (int): The pretraining TP value. Default is 1.
            slow_but_exact (bool): Determines if the method should prioritize accuracy over speed. Default is False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.slow_but_exact = slow_but_exact

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

__all__ = ['BloomConfig']
