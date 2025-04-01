from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from ldm.modules.attention import SpatialTransformer
from ldm.util import exists


# substituição
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


# go
class AttentionPool2d(nn.Module):
    """
    adaptado de clip: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None
    ):
        super().__init__()

        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape

        x = x.reshape(b, c, -1) # nc(hw)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1) # nc(hw + 1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype) # nc(hw + 1)

        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)

        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    qualquer módulo onde forward() toma embeddings de passo de tempo como um segundo argumento.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        aplica o módulo a `x` dado `emb` embeddings de passo de tempo.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    um módulo sequencial que passa embeddings de passo de tempo para os filhos que
    suporta isso como um input extra.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)

        return x


class Upsample(nn.Module):
    """
    um módulo de upsampling com uma convolução opcional.
    :param channels: canais na entrada e saída.
    :param use_conv: um bool que determina se uma convolução é aplicada.
    :param dims: determina se o sinal é 1d, 2d ou 3d. se 3d, então
                 upsampling ocorre nas duas dimensões internas.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels

        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x

class TransposedUpsample(nn.Module):
    """
    um upsampling 2x aprendido sem padding
    """

    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    um módulo de downsampling com uma convolução opcional.
    :param channels: canais na entrada e saída.
    :param use_conv: um bool que determina se uma convolução é aplicada.
    :param dims: determina se o sinal é 1d, 2d ou 3d. se 3d, então
                 downsampling ocorre nas duas dimensões internas.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        stride = 2 if dims != 3 else (1, 2, 2)

        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels

            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels

        return self.op(x)


class ResBlock(TimestepBlock):
    """
    um bloco residual que pode opcionalmente alterar o número de canais.
    :param channels: o número de canais de entrada.
    :param emb_channels: o número de canais de embedding de passo de tempo.
    :param dropout: a taxa de dropout.
    :param out_channels: se especificado, o número de canais de saída.
    :param use_conv: se true e out_channels está especificado, use uma convolução espacial
        convolução em vez de uma convolução 1x1 menor para alterar os canais
        na conexão skip.
    :param dims: determina se o sinal é 1d, 2d ou 3d.
    :param use_checkpoint: se true, use gradient checkpointing neste módulo.
    :param up: se true, use este bloco para upsampling.
    :param down: se true, use este bloco para downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False
    ):
        super().__init__()

        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),

            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels
            )
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),

            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            )
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        aplica o bloco a um tensor, condicionado em um embedding de passo de tempo.
        :param x: um tensor [n x c x ...] de características.
        :param emb: um tensor [n x emb_channels] de embeddings de passo de tempo.
        :return: um tensor [n x c x ...] de saídas.
        """

        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)

            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    um bloco de atenção que permite que as posições espaciais sejam atendidas a si mesmas.
    originalmente portado de:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False
    ):
        super().__init__()

        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v canais {channels} não são divisíveis por num_head_channels {num_head_channels}"

            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)

        if use_new_attention_order:
            # divide qkv antes de dividir as cabeças
            self.attention = QKVAttention(self.num_heads)
        else:
            # divide as cabeças antes de dividir qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)
        # return pt_checkpoint(self._forward, x) # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))

        h = self.attention(qkv)
        h = self.proj_out(h)

        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    um contador para o pacote `thop` para contar as operações em uma
    operação de atenção.
    destinado a ser usado como:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops}
        )
    """

    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))

    # realizamos dois matmuls com o mesmo número de ops.
    #
    # o primeiro calcula a matriz de pesos, o segundo calcula a combinação dos vetores
    # de valores.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    um módulo que executa a atenção qkv. corresponde ao qkvattention legado + modelagem de cabeças de entrada/saída
    """

    def __init__(self, n_heads):
        super().__init__()

        self.n_heads = n_heads

    def forward(self, qkv):
        """
        aplica a atenção qkv.
        :param qkv: um tensor [n x (h * 3 * c) x t] de qs, ks, e vs.
        :return: um tensor [n x (h * c) x t] após a atenção.
        """

        bs, width, length = qkv.shape

        assert width % (3 * self.n_heads) == 0

        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        ) # mais estável com f16 do que dividir depois

        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    um módulo que executa a atenção qkv e divide em uma ordem diferente.
    """

    def __init__(self, n_heads):
        super().__init__()

        self.n_heads = n_heads

    def forward(self, qkv):
        """
        aplica a atenção qkv.
        :param qkv: um tensor [N x (3 * H * C) x T] de qs, ks, e vs.
        :return: um tensor [N x (H * C) x T] após a atenção.
        """

        bs, width, length = qkv.shape

        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = th.einsum(
            "bct,bcs->bts",

            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        ) # mais estável com f16 do que dividir depois

        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))

        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    o modelo unet completo com atenção e incorporação de passo de tempo.
    :param in_channels: canais na entrada tensor.
    :param model_channels: contagem base de canais para o modelo.
    :param out_channels: canais na saída tensor.
    :param num_res_blocks: número de blocos residuais por downsample.
    :param attention_resolutions: uma coleção de taxas de downsample em que
        atenção ocorrerá. pode ser um conjunto, lista ou tupla.
        por exemplo, se este contém 4, então na downsample 4x, atenção
        será usada.
    :param dropout: a taxa de dropout.
    :param channel_mult: multiplicador de canal para cada nível do UNet.
    :param conv_resample: se True, use convoluções aprendidas para upsampling e
        downsampling.
    :param dims: determina se o sinal é 1d, 2d ou 3d.
    :param num_classes: se especificado (como um int), então este modelo será
        class-conditional com `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing para reduzir o uso de memória.
    :param num_heads: o número de cabeças de atenção em cada camada de atenção.
    :param num_heads_channels: se especificado, ignore num_heads e use
                               a fixed channel width per attention head.
    :param num_heads_upsample: funciona com num_heads para definir um número
                               diferente de cabeças para upsampling. Deprecated.
    :param use_scale_shift_norm: use um mecanismo de condicionamento como FiLM.
    :param resblock_updown: use blocos residuais para up/downsampling.
    :param use_new_attention_order: use uma ordem de atenção diferente para potencialmente
                                    aumentar a eficiência.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False, # suporte para transformador personalizado
        transformer_depth=1,           # suporte para transformador personalizado
        context_dim=None,              # suporte para transformador personalizado
        n_embed=None,                  # suporte para previsão de ids discretos em um código de primeiro estágio vq
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'você esqueceu de incluir a dimensão do seu condicionamento de atenção cruzada...'

        if context_dim is not None:
            assert use_spatial_transformer, 'você esqueceu de usar o transformador espacial para seu condicionamento de atenção cruzada...'
            
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'é necessário definir num_heads ou num_head_channels'

        if num_head_channels == -1:
            assert num_heads != -1, 'é necessário definir num_heads ou num_head_channels'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("forneça num_res_blocks como um int (globalmente constante) ou "
                                 "como uma lista/tupla (por nível) com o mesmo comprimento que channel_mult")
            
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # deve ser uma lista de booleanos, indicando se desabilitar a atenção própria no transformerblocks ou não
            
            assert len(disable_self_attentions) == len(channel_mult)

        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            
            print(f"o construtor de unetmodel recebeu num_attention_blocks={num_attention_blocks}. "
                  f"essa opção tem prioridade menor que attention_resolutions {attention_resolutions}, "
                  f"i.e., em casos onde num_attention_blocks[i] > 0 mas 2**i não está em attention_resolutions, "
                  f"atenção ainda não será definida.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("configurando a camada de incorporação linear c_adm")

                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = model_channels

        input_block_chans = [model_channels]

        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]

                ch = mult * model_channels

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch

                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )

                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )

                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),

            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order
            ) if not use_spatial_transformer else SpatialTransformer( # sempre usa uma auto-atenção
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )

        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()

                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]

                ch = model_channels * mult

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )

                if level and i == self.num_res_blocks[level]:
                    out_ch = ch

                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True
                        )

                        if resblock_updown
                        else
                            Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )

                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1))
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1)
            
            # nn.LogSoftmax(dim=1) # muda para cross_entropy e produzir logits não normalizados
        )

    def convert_to_fp16(self):
        """
        converte o torso do modelo para float16.
        """

        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        converte o torso do modelo para float32.
        """

        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        aplica o modelo a um lote de entrada.
        :param x: um tensor [n x c x ...] de entradas.
        :param timesteps: um tensor 1-d de passos de tempo.
        :param context: condicionamento plugado via crossattn
        :param y: um tensor [n] de rótulos, se condicional à classe.
        :return: um tensor [n x c x ...] de saídas.
        """

        assert (y is not None) == (
            self.num_classes is not None
        ), "deve especificar y se e somente se o modelo for condicional à classe"

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]

            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)