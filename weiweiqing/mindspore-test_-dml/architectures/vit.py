from mindspore import nn, ops
import model
import mindspore as ms
from . import register_arc


@register_arc
class Vit(nn.Cell):
	def __init__(self, opt, weight_file_path):
		super().__init__()
		temp_net = model.vit.ViT(num_classes=1000)

		params_dict = ms.load_checkpoint(weight_file_path)
		ms.load_param_into_net(temp_net, params_dict)

		self.vit_model = temp_net

		# replace dense layer
		n_features = self.vit_model.dense.in_channels
		self.vit_model.dense = nn.Dense(in_channels=n_features, out_channels=opt.embed_dim)

		self.pool = nn.AdaptiveAvgPool1d(output_size=1)
		self.multi_loss = opt.multi_loss

	def construct(self, x):

		# region from vit

		# embed & add class tokens & position encoding
		x = self.vit_model.patch_embedding(x)  # [b, n, vit_embed_dim] b-batch_size, n-num_patches
		cls_tokens = ops.tile(self.vit_model.cls_token.astype(x.dtype), (x.shape[0], 1, 1))  # [b, 1, vit_embed_dim]
		x = ops.concat((cls_tokens, x), axis=1)  # [b, n+1, vit_embed_dim]
		x += self.vit_model.pos_embedding  # [b, n+1, vit_embed_dim] + [1, n + 1, vit_embed_dim] => [b, n+1, vit_embed_dim]
		x = self.vit_model.pos_dropout(x)  # x: [batch_size, num_patches+1, vit_embed_dim] note, +1: class token

		x = self.vit_model.transformer(x)  # [b, n+1, vit_embed_dim]
		x = self.vit_model.norm(x)  # LayerNorm(vit_embed_dim)

		# endregion

		# last linear
		x = self.vit_model.dense(x)  # [b, n+1, embed_dim]
		x = x.transpose(0, 2, 1)  # [b, embed_dim, n+1]
		cls, patch = ops.split(x, [1, 196], axis=-1)  # cls: [b, embed_dim, 1]  patch: [b, embed_dim, 16*16]

		cls = cls.view(cls.shape[0], -1)  # [b, embed_dim]

		patch = self.pool(patch)  # [b, embed_dim, 1]
		patch = patch.view(patch.shape[0], -1)  # [b, embed_dim]

		# cls_embedding = ops.L2Normalize(axis=-1)(cls)
		# patch_embedding = ops.L2Normalize(axis=-1)(patch)

		if self.multi_loss:
			return ops.L2Normalize(axis=-1)(cls), ops.L2Normalize(axis=-1)(patch)   # [b, embed_dim], [b, embed_dim]
			# return cls_embedding, patch_embedding
		else:
			return ops.L2Normalize(axis=-1)(cls)  # [b, embed_dim]
			# return cls_embedding

