# fonte original:
# https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/attention_torch.py
#
# licença:
# mit
#
# crétidos
# amin rezaei (autor original)
# alex birch (algoritmo otimizado para tensors 3d)
#
# a autoatenção não precisa de memória o(n2):
# https://arxiv.org/abs/2112.05682v2

from functools import partial
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, NamedTuple, Protocol, List

from torch import Tensor
from typing import List


def dynamic_slice(
    x: Tensor,
    starts: List[int],
    sizes: List[int],
) -> Tensor:
    slicing = [slice(start, start + size) for start, size in zip(starts, sizes)]

    return x[slicing]


class AttnChunk(NamedTuple):
    exp_values: Tensor
    exp_weights_sum: Tensor
    max_score: Tensor

class SummarizeChunk(Protocol):
    @staticmethod
    def __call__(
        query: Tensor,
        key_t: Tensor,
        value: Tensor,
    ) -> AttnChunk: ...

class ComputeQueryChunkAttn(Protocol):
    @staticmethod
    def __call__(
        query: Tensor,
        key_t: Tensor,
        value: Tensor,
    ) -> Tensor: ...

def _summarize_chunk(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    scale: float,
) -> AttnChunk:
    attn_weights = torch.baddbmm(
        torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key_t,
        alpha=scale,
        beta=0,
    )

    max_score, _ = torch.max(attn_weights, -1, keepdim=True)
    max_score = max_score.detach()

    exp_weights = torch.exp(attn_weights - max_score)
    exp_values = torch.bmm(exp_weights, value)

    max_score = max_score.squeeze(-1)

    return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)

def _query_chunk_attention(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    summarize_chunk: SummarizeChunk,
    kv_chunk_size: int,
) -> Tensor:
    batch_x_heads, k_channels_per_head, k_tokens = key_t.shape
    _, _, v_channels_per_head = value.shape

    def chunk_scanner(chunk_idx: int) -> AttnChunk:
        key_chunk = dynamic_slice(
            key_t,
            (0, 0, chunk_idx),
            (batch_x_heads, k_channels_per_head, kv_chunk_size)
        )

        value_chunk = dynamic_slice(
            value,
            (0, chunk_idx, 0),
            (batch_x_heads, kv_chunk_size, v_channels_per_head)
        )

        return summarize_chunk(query, key_chunk, value_chunk)

    chunks: List[AttnChunk] = [
        chunk_scanner(chunk) for chunk in torch.arange(0, k_tokens, kv_chunk_size)
    ]

    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))
    chunk_values, chunk_weights, chunk_max = acc_chunk

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)

    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)

    return all_values / all_weights

# todo: refatora crossattention#get_attention_scores para compartilhar código com isso
def _get_attention_scores_no_kv_chunking(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    scale: float,
) -> Tensor:
    attn_scores = torch.baddbmm(
        torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key_t,
        alpha=scale,
        beta=0
    )

    attn_probs = attn_scores.softmax(dim=-1)

    del attn_scores

    hidden_states_slice = torch.bmm(attn_probs, value)

    return hidden_states_slice

class ScannedChunk(NamedTuple):
    chunk_idx: int
    attn_chunk: AttnChunk

def efficient_dot_product_attention(
    query: Tensor,
    key_t: Tensor,
    value: Tensor,
    query_chunk_size=1024,
    kv_chunk_size: Optional[int] = None,
    kv_chunk_size_min: Optional[int] = None,
    use_checkpoint=True
):
    """
    calcula a atenção eficiente do produto escalar dada a consulta, chave transposta e valor.
    esta é uma versão eficiente da atenção apresentada em:
    https://arxiv.org/abs/2112.05682v2 que vem com requisitos de memória o(sqrt(n)).
    args:
        query: consultas para calcular atenção com forma de
          `[batch * num_heads, tokens, channels_per_head]`.
        key_t: chaves para calcular a atenção com a forma de
          `[batch * num_heads, channels_per_head, tokens]`.
        value: valores a serem usados ​​na atenção com forma de
          `[batch * num_heads, tokens, channels_per_head]`.
        query_chunk_size: int: tamanho dos blocos de consulta
        kv_chunk_size: optional[int]: tamanho dos blocos chave/valor. caso none: padrões para sqrt(key_tokens)
        kv_chunk_size_min: optional[int]: tamanho mínimo do bloco chave/valor. considerado somente quando kv_chunk_size é Nenhum. altera `sqrt(key_tokens)` para `max(sqrt(key_tokens), kv_chunk_size_min)`, para garantir que nossos tamanhos de blocos não fiquem muito pequenos (blocos menores = mais blocos = menos trabalho simultâneo feito).
        use_checkpoint: bool: se deve usar checkpointing (recomendado: true para treinamento, false para inferência)
    retorna:
        saída de forma `[batch * num_heads, query_tokens, channels_per_head]`.
    """
    
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, _, k_tokens = key_t.shape
    scale = q_channels_per_head ** -0.5

    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    def get_query_chunk(chunk_idx: int) -> Tensor:
        return dynamic_slice(
            query,
            (0, chunk_idx, 0),
            (batch_x_heads, min(query_chunk_size, q_tokens), q_channels_per_head)
        )
    
    summarize_chunk: SummarizeChunk = partial(_summarize_chunk, scale=scale)
    summarize_chunk: SummarizeChunk = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    
    compute_query_chunk_attn: ComputeQueryChunkAttn = partial(
        _get_attention_scores_no_kv_chunking,
        scale=scale
    ) if k_tokens <= kv_chunk_size else (
        # caminho rápido para quando há apenas 1 pedaço de chave-valor por pedaço de consulta
        partial(
            _query_chunk_attention,
            kv_chunk_size=kv_chunk_size,
            summarize_chunk=summarize_chunk,
        )
    )

    if q_tokens <= query_chunk_size:
        # caminho rápido para quando há apenas 1 pedaço de consulta
        return compute_query_chunk_attn(
            query=query,
            key_t=key_t,
            value=value,
        )
    
    res = torch.cat([
        compute_query_chunk_attn(
            query=get_query_chunk(i * query_chunk_size),
            key_t=key_t,
            value=value,
        ) for i in range(math.ceil(q_tokens / query_chunk_size))
    ], dim=1)
    
    return res