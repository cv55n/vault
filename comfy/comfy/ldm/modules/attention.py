from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

fromfrom ldm.modules.diffusionmodules.util import checkpoint
from .sub_quadratic_attention import efficient_dot_product_attention


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# manuseio de precisão crossattn
import os

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)

    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)

        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()

        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    zera os parâmetros de um módulo e o retorna.
    """

    for p in module.parameters():
        p.detach().zero_()

    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.norm = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x

        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # calcula a atenção
        b,c,h,w = q.shape

        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')

        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # atenta aos valores
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttentionBirchSan(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        query = self.to_q(x)
        context = default(context, x)
        key = self.to_k(context)
        value = self.to_v(context)
        del context, x

        query = query.unflatten(-1, (self.heads, -1)).transpose(1,2).flatten(end_dim=1)
        key_t = key.transpose(1,2).unflatten(1, (self.heads, -1)).flatten(end_dim=1)

        del key

        value = value.unflatten(-1, (self.heads, -1)).transpose(1,2).flatten(end_dim=1)

        dtype = query.dtype

        # if self.upcast_attention:
        #     query = query.float()
        #     key_t = key_t.float()

        bytes_per_token = torch.finfo(query.dtype).bits//8
        batch_x_heads, q_tokens, _ = query.shape
        _, _, k_tokens = key_t.shape
        qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

        stats = torch.cuda.memory_stats(query.device)

        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']

        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        chunk_threshold_bytes = mem_free_torch * 0.5 # usar apenas isso parece funcionar melhor no amd

        kv_chunk_size_min = None

        query_chunk_size_x = 1024 * 4

        kv_chunk_size_min_x = None
        kv_chunk_size_x = (int((chunk_threshold_bytes // (batch_x_heads * bytes_per_token * query_chunk_size_x)) * 1.2) // 1024) * 1024
        
        if kv_chunk_size_x < 1024:
            kv_chunk_size_x = None

        if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
            # o grande matmul se encaixa no nosso limite de memória; faça tudo em 1
            # pedaço, ou seja, envie-o pelo caminho rápido não dividido

            query_chunk_size = q_tokens
            kv_chunk_size = k_tokens
        else:
            query_chunk_size = query_chunk_size_x
            kv_chunk_size = kv_chunk_size_x
            kv_chunk_size_min = kv_chunk_size_min_x

        hidden_states = efficient_dot_product_attention(
            query,
            key_t,
            value,

            query_chunk_size=query_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min=kv_chunk_size_min,
            use_checkpoint=self.training,
        )

        hidden_states = hidden_states.to(dtype)

        hidden_states = hidden_states.unflatten(0, (-1, self.heads)).transpose(1,2).flatten(start_dim=2)

        out_proj, dropout = self.to_out

        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        return hidden_states


class CrossAttentionDoggettx(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)

        k_in = self.to_k(context)
        v_in = self.to_v(context)

        del context, x

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

        stats = torch.cuda.memory_stats(q.device)

        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        gb = 1024 ** 3

        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier

        steps = 1


        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

            # print(f"tamanho do tensor esperado: {tensor_size/gb:0.1f}gb, cuda livre: {mem_free_cuda/gb:0.1f}gb "
            #      f"torch livre:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} passos:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64

            raise RuntimeError(f'memória insuficiente, use uma resolução mais baixa (máx. aprox. {max_res}x{max_res}). '
                               f'necessário: {mem_required/64/gb:0.1f}gb livre, possui: {mem_free_total/gb:0.1f}gb livre')

        # print("steps", steps, mem_required, mem_free_total, modifier, q.element_size(), tensor_size)
        
        first_op_done = False
        cleared_cache = False

        while True:
            try:
                slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]

                for i in range(0, q.shape[1], slice_size):
                    end = i + slice_size

                    if _ATTN_PRECISION =="fp32":
                        with torch.autocast(enabled=False, device_type = 'cuda'):
                            s1 = einsum('b i d, b j d -> b i j', q[:, i:end].float(), k.float()) * self.scale
                    else:
                        s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale
                    
                    first_op_done = True

                    s2 = s1.softmax(dim=-1)

                    del s1

                    r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)

                    del s2

                break
            except torch.cuda.OutOfMemoryError as e:
                if first_op_done == False:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                    if cleared_cache == False:
                        cleared_cache = True

                        print("erro de falta de memória, esvaziando o cache e tentando novamente")

                        continue

                    steps *= 2

                    if steps > 64:
                        raise e
                    
                    print("erro de falta de memória, aumentando os passos e tentando novamente", steps)
                else:
                    raise e

        del q, k, v

        r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
        del r1

        return self.to_out(r2)

class OriginalCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # força a conversão para fp32 para evitar estouro
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # atenção
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)

class CrossAttention(CrossAttentionDoggettx):
    pass

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        print(f"configurando {self.__class__.__name__}. dim da query é {query_dim}, context_dim é {context_dim} e está utilizando "
              f"{heads} cabeças.")
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape

        q, k, v = map(
            lambda t: t.unsqueeze(3)

            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # calcula a atenção
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)

            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention, # atenção vanilla
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()

        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None) # é self-attention caso não seja self.disable_self_attn
        
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout) # é self-attention se o contexto não for nenhum
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x

        x = self.ff(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):
    """
    bloco transformador para dados semelhantes a imagens.
    primeiro, projeta a entrada (incorporação) e a modela novamente para b, t, d.
    em seguida, aplica a ação padrão do transformador.
    por fim, modelando a imagem novamente.
    novo: use_linear para mais eficiência em vez das conversões 1x1
    """

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()

        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))

        self.use_linear = use_linear

    def forward(self, x, context=None):
        # nota: se nenhum contexto for fornecido, a atenção cruzada se torna, por padrão, autoatenção
        if not isinstance(context, list):
            context = [context]

        b, c, h, w = x.shape

        x_in = x
        x = self.norm(x)

        if not self.use_linear:
            x = self.proj_in(x)

        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])

        if self.use_linear:
            x = self.proj_out(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        if not self.use_linear:
            x = self.proj_out(x)
            
        return x + x_in