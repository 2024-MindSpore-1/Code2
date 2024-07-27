# coding=utf-8
# Copyright 2022 Snapchat Research and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore EfficientFormer model."""
# pylint: disable=consider-using-in
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import Normal, initializer

from mindnlp.utils import logging, ModelOutput
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel

from .configuration_efficientformer import EfficientFormerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "snap-research/efficientformer-l1-300"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 448]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformer-l1-300"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "snap-research/efficientformer-l1-300",
    # See all EfficientFormer models at https://huggingface.co/models?filter=efficientformer
]

class EfficientFormerPatchEmbeddings(nn.Cell):
    """
    This class performs downsampling between two stages. For the input tensor with the shape [batch_size, num_channels,
    height, width] it produces output tensor with the shape [batch_size, num_channels, height/stride, width/stride]
    """

    def __init__(self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool = True):
        super().__init__()
        self.num_channels = num_channels

        self.projection = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=config.downsample_patch_size,
            stride=config.downsample_stride,
            padding=config.downsample_pad,
            has_bias=True,
            pad_mode="pad"
        )
        self.norm = nn.BatchNorm2d(embed_dim, eps=config.batch_norm_eps) if apply_norm else nn.Identity()

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values)
        embeddings = self.norm(embeddings)

        return embeddings


class EfficientFormerSelfAttention(nn.Cell):
    def __init__(self, dim: int, key_dim: int, num_heads: int, attention_ratio: int, resolution: int):
        super().__init__()

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.scale = key_dim**-0.5
        self.total_key_dim = key_dim * num_heads
        self.expanded_key_dim = int(attention_ratio * key_dim)
        self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2
        self.qkv = nn.Dense(dim, hidden_size)
        self.projection = nn.Dense(self.total_expanded_key_dim, dim)
        points = list(itertools.product(range(resolution), range(resolution)))
        num_points = len(points)
        attention_offsets = {}
        idxs = []
        for point_1 in points:
            for point_2 in points:
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = Parameter(ops.zeros(num_heads, len(attention_offsets)))
        self.attention_bias_idxs = mindspore.Tensor(idxs).view(num_points, num_points)

    def construct(self, hidden_states: mindspore.Tensor, output_attentions: bool = False) -> Tuple[mindspore.Tensor]:
        batch_size, sequence_length, num_channels = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query_layer, key_layer, value_layer = qkv.reshape(batch_size, sequence_length, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.expanded_key_dim], axis=3
        )
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        # set `model.to(torch_device)` won't change `self.ab.device`, if there is no follow-up `train` or `eval` call.
        # Let's do it manually here, so users won't have to do this everytime.
        attention_probs = (ops.matmul(query_layer, key_layer.swapaxes(-2, -1))) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs]
        )

        attention_probs = ops.softmax(attention_probs,axis=-1)

        context_layer = ops.matmul(attention_probs, value_layer).swapaxes(1, 2)
        context_layer = context_layer.reshape(batch_size, sequence_length, self.total_expanded_key_dim)
        context_layer = self.projection(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class EfficientFormerConvStem(nn.Cell):
    def __init__(self, config: EfficientFormerConfig, out_channels: int):
        super().__init__()

        self.convolution1 = nn.Conv2d(config.num_channels, out_channels // 2, kernel_size=3, stride=2, padding=1,has_bias=True,pad_mode="pad")
        self.batchnorm_before = nn.BatchNorm2d(out_channels // 2, eps=config.batch_norm_eps)

        self.convolution2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1,has_bias=True,pad_mode="pad")
        self.batchnorm_after = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)

        self.activation = nn.ReLU()

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        features = self.batchnorm_before(self.convolution1(pixel_values))
        features = self.activation(features)
        features = self.batchnorm_after(self.convolution2(features))
        features = self.activation(features)

        return features


class EfficientFormerPooling(nn.Cell):
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, pad_mode="pad", padding=pool_size // 2, count_include_pad=False)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        output = self.pool(hidden_states) - hidden_states
        return output


class EfficientFormerDenseMlp(nn.Cell):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear_in = nn.Dense(in_features, hidden_features)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.linear_out = nn.Dense(hidden_features, out_features)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.linear_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class EfficientFormerConvMlp(nn.Cell):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.convolution1 = nn.Conv2d(in_features, hidden_features, 1, has_bias=True)
        self.activation = ACT2FN[config.hidden_act]
        self.convolution2 = nn.Conv2d(hidden_features, out_features, 1, has_bias=True)
        self.dropout = nn.Dropout(p=drop)

        self.batchnorm_before = nn.BatchNorm2d(hidden_features, eps=config.batch_norm_eps)
        self.batchnorm_after = nn.BatchNorm2d(out_features, eps=config.batch_norm_eps)

    def construct(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.batchnorm_before(hidden_state)

        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        hidden_state = self.batchnorm_after(hidden_state)
        hidden_state = self.dropout(hidden_state)

        return hidden_state


def drop_path(input: mindspore.Tensor, drop_prob: float = 0.0, training: bool = False) -> mindspore.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + ops.rand(shape, dtype=input.dtype)
    random_tensor.floor()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class EfficientFormerDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class EfficientFormerFlat(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, hidden_states: mindspore.Tensor) -> Tuple[mindspore.Tensor]:
        hidden_states = hidden_states.flatten(start_dim=2).swapaxes(1, 2)
        return hidden_states


class EfficientFormerMeta3D(nn.Cell):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0):
        super().__init__()

        self.token_mixer = EfficientFormerSelfAttention(
            dim=config.dim,
            key_dim=config.key_dim,
            num_heads=config.num_attention_heads,
            attention_ratio=config.attention_ratio,
            resolution=config.resolution,
        )

        self.layernorm1 = nn.LayerNorm([dim], epsilon=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm([dim], epsilon=config.layer_norm_eps)

        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim)

        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = Parameter(config.layer_scale_init_value * ops.ones((dim)), requires_grad=True)
            self.layer_scale_2 = Parameter(config.layer_scale_init_value * ops.ones((dim)), requires_grad=True)

    def construct(self, hidden_states: mindspore.Tensor, output_attentions: bool = False) -> Tuple[mindspore.Tensor]:
        self_attention_outputs = self.token_mixer(self.layernorm1(hidden_states), output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.use_layer_scale:
            layer_output = hidden_states + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0) * attention_output
            )
            layer_output = layer_output + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.layernorm2(layer_output))
            )
        else:
            layer_output = hidden_states + self.drop_path(attention_output)
            layer_output = layer_output + self.drop_path(self.mlp(self.layernorm2(layer_output)))

        outputs = (layer_output,) + outputs

        return outputs


class EfficientFormerMeta3DLayers(nn.Cell):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:-1]))
            for block_idx in range(config.num_meta3d_blocks)
        ]
        self.blocks = nn.CellList(
            [EfficientFormerMeta3D(config, config.hidden_sizes[-1], drop_path=drop_path) for drop_path in drop_paths]
        )

    def construct(self, hidden_states: mindspore.Tensor, output_attentions: bool = False) -> Tuple[mindspore.Tensor]:
        all_attention_outputs = () if output_attentions else None

        for layer_module in self.blocks:
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            hidden_states = layer_module(hidden_states, output_attentions)

            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)

        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs

        return hidden_states


class EfficientFormerMeta4D(nn.Cell):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0):
        super().__init__()
        pool_size = config.pool_size if config.pool_size is not None else 3
        self.token_mixer = EfficientFormerPooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientFormerConvMlp(
            config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob
        )

        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = Parameter(config.layer_scale_init_value * ops.ones((dim)), requires_grad=True)
            self.layer_scale_2 = Parameter(config.layer_scale_init_value * ops.ones((dim)), requires_grad=True)

    def construct(self, hidden_states: mindspore.Tensor) -> Tuple[mindspore.Tensor]:
        outputs = self.token_mixer(hidden_states)

        if self.use_layer_scale:
            layer_output = hidden_states + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * outputs)

            layer_output = layer_output + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(layer_output)
            )
        else:
            layer_output = hidden_states + self.drop_path(outputs)
            layer_output = layer_output + self.drop_path(self.mlp(layer_output))

        return layer_output


class EfficientFormerMeta4DLayers(nn.Cell):
    def __init__(self, config: EfficientFormerConfig, stage_idx: int):
        super().__init__()
        num_layers = (
            config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        )
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)
        ]

        self.blocks = nn.CellList(
            [
                EfficientFormerMeta4D(config, config.hidden_sizes[stage_idx], drop_path=drop_path)
                for drop_path in drop_paths
            ]
        )

    def construct(self, hidden_states: mindspore.Tensor) -> Tuple[mindspore.Tensor]:
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class EfficientFormerIntermediateStage(nn.Cell):
    def __init__(self, config: EfficientFormerConfig, index: int):
        super().__init__()
        self.meta4D_layers = EfficientFormerMeta4DLayers(config, index)

    def construct(self, hidden_states: mindspore.Tensor) -> Tuple[mindspore.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states)
        return hidden_states


class EfficientFormerLastStage(nn.Cell):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        self.meta4D_layers = EfficientFormerMeta4DLayers(config, -1)
        self.flat = EfficientFormerFlat()
        self.meta3D_layers = EfficientFormerMeta3DLayers(config)

    def construct(self, hidden_states: mindspore.Tensor, output_attentions: bool = False) -> Tuple[mindspore.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states)
        hidden_states = self.flat(hidden_states)
        hidden_states = self.meta3D_layers(hidden_states, output_attentions)

        return hidden_states


class EfficientFormerEncoder(nn.Cell):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        self.config = config
        num_intermediate_stages = len(config.depths) - 1
        downsamples = [
            config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1]
            for i in range(num_intermediate_stages)
        ]
        intermediate_stages = []

        for i in range(num_intermediate_stages):
            intermediate_stages.append(EfficientFormerIntermediateStage(config, i))
            if downsamples[i]:
                intermediate_stages.append(
                    EfficientFormerPatchEmbeddings(config, config.hidden_sizes[i], config.hidden_sizes[i + 1])
                )

        self.intermediate_stages = nn.CellList(intermediate_stages)
        self.last_stage = EfficientFormerLastStage(config)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions)

        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)

        if not return_dict:
            return tuple(v for v in [layer_output[0], all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=layer_output[0],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class EfficientFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientFormerConfig
    base_model_prefix = "efficientformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, cell: nn.Cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, nn.Conv2d)):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range), cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))



class EfficientFormerModel(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)
        self.config = config
        _no_split_modules = ["EfficientFormerMeta4D"]

        self.patch_embed = EfficientFormerConvStem(config, config.hidden_sizes[0])
        self.encoder = EfficientFormerEncoder(config)
        self.layernorm = nn.LayerNorm([config.hidden_sizes[-1]], epsilon=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.patch_embed(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class EfficientFormerForImageClassification(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = EfficientFormerModel(config)

        # Classifier head
        self.classifier = (
            nn.Dense(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()


    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output.mean(-2))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == mindspore.int64 or labels.dtype == mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class EfficientFormerForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`EfficientFormerForImageClassificationWithTeacher`].

    Args:
        logits (`mindspore.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`mindspore.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`mindspore.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: mindspore.Tensor = None
    cls_logits: mindspore.Tensor = None
    distillation_logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None

class EfficientFormerForImageClassificationWithTeacher(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = EfficientFormerModel(config)

        # Classifier head
        self.classifier = nn.Dense(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        # Distillation head
        self.distillation_classifier = (
            nn.Dense(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, EfficientFormerForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        cls_logits = self.classifier(sequence_output.mean(-2))
        distillation_logits = self.distillation_classifier(sequence_output.mean(-2))

        # during inference, return the average of both classifier predictions
        logits = (cls_logits + distillation_logits) / 2

        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        return EfficientFormerForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
        "EfficientFormerForImageClassification",
        "EfficientFormerForImageClassificationWithTeacher",
        "EfficientFormerModel",
        "EfficientFormerPreTrainedModel",
           ]
