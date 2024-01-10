# ------- from mindspore tutorial ------------------

from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Normal
import mindspore as ms
from typing import Optional
from . import ModelUtil


# Q * transpose(K) * V
class Attention(nn.Cell):
	def __init__(
			self,
			dim: int,
			num_heads: int = 8,
			drop_prop: float = 0,
			attention_drop_prob: float = 0,
	):
		super().__init__()

		self.num_heads = num_heads
		assert dim % num_heads == 0, "dim must be divisible by num_heads"
		head_dim = dim // num_heads
		self.scale = Tensor(head_dim ** -0.5)

		self.qkv = nn.Dense(dim, dim * 3)
		self.attn_drop = nn.Dropout(p=attention_drop_prob)

		self.projection = nn.Dense(dim, dim)  # for last projection
		self.out_drop = nn.Dropout(p=drop_prop)

	def construct(self, x):
		b, n, c = x.shape  # b - batch, n - number of vectors, c - dim / channels

		# get Q K V
		qkv = self.qkv(x)
		qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))  # [b, n, 3(Q/K/V), h, dim(c)/h]
		qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # [3, b, h, n, c/h]
		q, k, v = ops.unstack(qkv, axis=0)  # Q K V both are [b, h, n, c/h]

		# [n, c/h] is the unit of attention
		attn = ops.BatchMatMul(transpose_b=True)(q, k)
		attn = ops.mul(attn, self.scale)
		attn = ops.softmax(attn, axis=-1)
		attn = self.attn_drop(attn)

		out = ops.BatchMatMul()(attn, v)

		out = ops.transpose(out, (0, 2, 1, 3))   # [b, n, h, c/h]
		out = ops.reshape(out, (b, n, c))  # [b, n, c]

		out = self.projection(out)
		out = self.out_drop(out)

		return out


class FeedForward(nn.Cell):
	def __init__(
			self,
			in_features: int,
			hidden_features: Optional[int] = None,
			out_features: Optional[int] = None,
			activation: nn.Cell = nn.GELU,
			drop_prop: float = 0,
	):
		super().__init__()
		hidden_features = hidden_features or in_features
		out_features = out_features or in_features

		self.dense1 = nn.Dense(in_features, hidden_features)
		self.activation = activation()
		self.dense2 = nn.Dense(hidden_features, out_features)
		self.dropout = nn.Dropout(p=drop_prop)

	def construct(self, x):
		x = self.dense1(x)
		x = self.activation(x)
		x = self.dense2(x)
		x = self.dropout(x)
		return x


class ResidualCell(nn.Cell):
	def __init__(self, cell):
		super().__init__()
		self.cell = cell

	def construct(self, x):
		return self.cell(x) + x


class TransformerEncoder(nn.Cell):
	def __init__(
			self,
			dim: int,
			num_layers: int,
			num_heads: int,
			ffn_hidden_features: int,
			drop_prob: float = 0,
			attention_drop_prob: float = 0,
			ffn_drop_prob: float = 0,
			activation: nn.Cell = nn.GELU,
			norm: nn.Cell = nn.LayerNorm
	):
		super().__init__()
		layers = []

		for _ in range(num_layers):
			normalization1 = norm((dim,))
			normalization2 = norm((dim,))
			attention = Attention(
				dim=dim,
				num_heads=num_heads,
				drop_prop=drop_prob,
				attention_drop_prob=attention_drop_prob
			)
			feedforward = FeedForward(
				in_features=dim,
				hidden_features=ffn_hidden_features,
				activation=activation,
				drop_prop=ffn_drop_prob,
			)

			layers.append(
				nn.SequentialCell([
					ResidualCell(nn.SequentialCell([normalization1, attention])),
					ResidualCell(nn.SequentialCell([normalization2, feedforward]))
				])
			)

		self.layers = nn.SequentialCell(layers)

	def construct(self, x):
		return self.layers(x)


class PatchEmbedding(nn.Cell):
	def __init__(
			self,
			image_size: int = 224,
			patch_size: int = 16,
			input_channels: int = 3,
	):
		super().__init__()
		self.embed_dim = patch_size * patch_size * input_channels
		self.num_patches = (image_size // patch_size) ** 2
		self.conv = nn.Conv2d(input_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)

	def construct(self, x):
		x = self.conv(x)
		b, c, h, w = x.shape
		x = ops.reshape(x, (b, c, h * w))
		x = ops.transpose(x, (0, 2, 1))
		return x


class ViT(nn.Cell):
	def __init__(
			self,
			num_classes: int,
			image_size: int = 224,
			input_channels: int = 3,
			patch_size: int = 16,

			num_heads: int = 12,
			attention_drop_prob: float = 0,
			drop_prob: float = 0,

			ffn_hidden_features: int = 3072,
			activation: nn.Cell = nn.GELU,
			ffn_drop_prob: float = 0,

			norm: nn.Cell = nn.LayerNorm,
			num_layers: int = 12,
	) -> None:
		super().__init__()

		self.patch_embedding = PatchEmbedding(
			image_size=image_size,
			patch_size=patch_size,
			input_channels=input_channels
		)

		embed_dim = self.patch_embedding.embed_dim
		num_patchers = self.patch_embedding.num_patches

		# class embedding?
		self.cls_token = ModelUtil.create_parameter(
			init_type=Normal(sigma=1.0),
			shape=(1, 1, embed_dim),
			dtype=ms.float32,
			name="cls",
			requires_grad=True
		)

		# position embedding
		self.pos_embedding = ModelUtil.create_parameter(
			init_type=Normal(sigma=1.0),
			shape=(1, num_patchers + 1, embed_dim),
			dtype=ms.float32,
			name="pos_embedding",
			requires_grad=True
		)

		self.pos_dropout = nn.Dropout(p=drop_prob)

		self.transformer = TransformerEncoder(
			dim=embed_dim,
			num_heads=num_heads,
			attention_drop_prob=attention_drop_prob,
			drop_prob=drop_prob,

			ffn_hidden_features=ffn_hidden_features,
			activation=activation,
			ffn_drop_prob=ffn_drop_prob,

			norm=norm,
			num_layers=num_layers
		)

		self.norm = norm((embed_dim,))
		self.dropout = nn.Dropout(p=drop_prob)
		self.dense = nn.Dense(embed_dim, num_classes)

	def construct(self, x):
		# embed & add class tokens & position encoding
		x = self.patch_embedding(x)
		cls_tokens = ops.tile(self.cls_token.astype(x.dtype), (x.shape[0], 1, 1))
		x = ops.concat((cls_tokens, x), axis=1)
		x += self.pos_embedding
		x = self.pos_dropout(x)  # x: [batch_size, num_patches+1, embed_dim] note, +1: class token

		x = self.transformer(x)
		x = self.norm(x)

		x = x[:, 0]
		x = self.dropout(x)
		x = self.dense(x)

		return x
