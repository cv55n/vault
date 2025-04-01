# adotado de:
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
#
# e:
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
#
# e:
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# obrigado

import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat

from ldm.util import instantiate_from_config


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )

        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]

        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' desconhecido.")
    
    return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps

        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'não existe um método de discretização ddim chamado "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps

    # adiciona um para obter os valores alfa finais corretos (os da primeira escala para os dados durante a amostragem)
    steps_out = ddim_timesteps + 1
    
    if verbose:
        print(f'passos de tempo selecionados para o amostrador ddim: {steps_out}')

    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # seleciona os alfas para calcular o cronograma de variância
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # de acordo com a fórmula fornecida em: https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

    if verbose:
        print(f'alfas selecionados para o amostrador ddim: a_t: {alphas}; a_(t-1): {alphas_prev}')

    print(f'para o valor escolhido de eta, que é {eta}, '
          f'isso resulta no seguinte cronograma sigma_t para o amostrador ddim {sigmas}')
        
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    cria um cronograma de beta que discretiza a função alpha_t_bar fornecida,
    que define o produto cumulativo de (1-beta) ao longo do tempo de t = [0,1].
    :param num_diffusion_timesteps: o número de betas a produzir.
    :param alpha_bar: uma lambda que toma um argumento t de 0 a 1 e
                      produz o produto cumulativo de (1-beta) até esse
                      parte do processo de difusão.
    :param max_beta: o máximo beta a usar; use valores menores que 1 para
                     evitar singularidades.
    """

    betas = []

    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps

        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)

    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def checkpoint(func, inputs, params, flag):
    """
    avalia uma função sem armazenar ativações intermediárias, permitindo
    redução de memória ao custo de extra computação no retrocesso.
    :param func: a função a ser avaliada.
    :param inputs: a sequência de argumentos a ser passada para `func`.
    :param params: uma sequência de parâmetros `func` depende, mas não
                   explicitamente toma como argumentos.
    :param flag: se false, desabilita o checkpointing de gradientes.
    """

    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)

        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # corrige um bug onde a primeira operação em run_function modifica o
            # armazenamento do tensor em lugar, o que não é permitido para tensors detach()'d
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]

            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True
        )
        
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors

        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    cria incorporações de passo de tempo sinusoidal.
    :param timesteps: um tensor 1d de n índices, um por elemento do lote.
                      esses podem ser fracionários.
    :param dim: a dimensão da saída.
    :param max_period: controla a frequência mínima dos embeddings.
    :return: um tensor [n x dim] de incorporações de posição.
    """

    if not repeat_only:
        half = dim // 2

        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)

    return embedding


def zero_module(module):
    """
    zera os parâmetros de um módulo e retorna ele.
    """

    for p in module.parameters():
        p.detach().zero_()

    return module


def scale_module(module, scale):
    """
    escala os parâmetros de um módulo e retorna ele.
    """

    for p in module.parameters():
        p.detach().mul_(scale)

    return module


def mean_flat(tensor):
    """
    calcula a média de todas as dimensões que não são de lote.
    """

    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    cria uma camada de normalização padrão.
    :param channels: número de canais de entrada.
    :return: um nn.module para normalização.
    """

    return GroupNorm32(32, channels)


# pytorch 1.7 tem silu, mas suportamos apenas o pytorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    cria um módulo de convolução 1d, 2d ou 3d.
    """

    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    
    raise ValueError(f"dimensões não suportadas: {dims}")


def linear(*args, **kwargs):
    """
    cria um módulo linear.
    """

    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    cria um módulo de agrupamento de médias 1d, 2d ou 3d.
    """

    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    
    raise ValueError(f"dimensões não suportadas: {dims}")


class HybridConditioner(nn.Module):
    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()

        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)

        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()