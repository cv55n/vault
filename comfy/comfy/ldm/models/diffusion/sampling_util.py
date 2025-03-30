import torch
import numpy as np


def append_dims(x, target_dims):
    """acrescenta dimensões ao final de um tensor até que ele tenha dimensões target_dims.
    de: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/utils.py"""

    dims_to_append = target_dims - x.ndim
    
    if dims_to_append < 0:
        raise ValueError(f'a entrada tem {x.ndim} dims mas target_dims é {target_dims}, o que é menos')
    
    return x[(...,) + (None,) * dims_to_append]


def norm_thresholding(x0, value):
    s = append_dims(x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value), x0.ndim)

    return x0 * (value / s)


def spatial_norm_thresholding(x0, value):
    # b c h w
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)

    return x0 * (value / s)