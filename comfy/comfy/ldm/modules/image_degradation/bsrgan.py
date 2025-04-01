# -*- coding: utf-8 -*-
"""
# ----------------------------------------
# super-resolução
# ----------------------------------------
"""

import numpy as np
import cv2
import torch

from functools import partial
import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
import albumentations

import ldm.modules.image_degradation.utils_image as util


def modcrop_np(img, sf):
    '''
    args:
        img: imagem numpy, xwh ou wxhxc
        sf: fator de escala
    retorna:
        imagem recortada
    '''

    w, h = img.shape[:2]
    im = np.copy(img)

    return im[:w - w % sf, :h - h % sf, ...]


"""
# ----------------------------------------
# núcleos gaussianos anisotrópicos
# ----------------------------------------
"""

def analytic_kernel(k):
    """calcula o kernel x4 a partir do kernel x2"""

    k_size = k.shape[0]

    # calcula o tamanho dos grãos grandes
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))

    # passa sobre o grão pequeno para preencher o grande
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k

    # corta as bordas do kernel grande para ignorar valores muito pequenos e aumentar o tempo de execução do sr
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]

    # normaliza para 1
    return cropped_big_k / cropped_big_k.sum()


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """gera um kernel gaussiano anisotrópico
    args:
        ksize : ex., 15, tamanho do kernel
        theta : [0,  pi], ângulo de rotação
        l1    : [0.1,50], escala de eigenvalues
        l2    : [0.1,l1], escala de eigenvalues
        se l1 = l2, obterá um kernel gaussiano isotrópico.
    retorna:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5

    k = np.zeros([size, size])

    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1

            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)

    return k


def shift_pixel(x, sf, upper_left=True):
    """desloca o pixel para super-resolução com diferentes fatores de escala
    args:
        x: wxhxc ou wxh
        sf: fator de escala
        upper_left: direção de deslocamento
    """

    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5

    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)

    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def blur(x, k):
    '''
    x: imagem, nxcxhxw
    k: kernel, nx1xhxw
    '''

    n, c = x.shape[:2]

    p1, p2 = (k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')

    k = k.repeat(1, c, 1, 1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])

    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n * c)
    x = x.view(n, c, x.shape[2], x.shape[3])

    return x


def gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
    """"
    # versão modificada de: https://github.com/assafshocher/BlindSR_dataset_generator
    #
    # min_var = 0.175 * sf # a variância do kernel gaussiano será amostrada entre min_var e max_var
    # max_var = 2.5 * sf
    """

    # define autovalores aleatórios (lambdas) e ângulo (theta) para a matriz cov
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)

    theta = np.random.rand() * np.pi # theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # define a matriz cov usando lambdas e theta
    LAMBDA = np.diag([lambda_1, lambda_2])

    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # define a posição de expectativa (deslocando kernel para imagem alinhada)
    MU = k_size // 2 - 0.5 * (scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # cria uma grade para o kernel gaussiano
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # calcula o gaussiano para cada pixel do kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)

    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # desloca o kernel para que esteja centralizado
    # raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # normaliza o kernel e retorna
    # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)

    return kernel


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]

    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma

    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    
    arg = -(x * x + y * y) / (2 * std * std)

    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0

    sumh = h.sum()

    if sumh != 0:
        h = h / sumh

    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])

    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)

    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = np.array(h)

    return h


def fspecial(filter_type, *args, **kwargs):
    '''
    código python de:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''

    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)


"""
# ----------------------------------------
# modelos de degradação
# ----------------------------------------
"""


def bicubic_degradation(x, sf=3):
    '''
    args:
        x: imagem hxwxc, [0, 1]
        sf: down-scale factor
    retorna:
        imagem lr com subamostragem bicúbica
    '''

    x = util.imresize_np(x, scale=1 / sf)
    
    return x


def srmd_degradation(x, k, sf=3):
    '''
    borrão + subamostragem bicúbica
    args:
        x: imagem hxwxc, [0, 1]
        k: hxw, dobro
        sf: fator de escala de subamostragem
    retorna:
        imagem lr com subamostragem bicúbica
    referência:
        @inproceedings{zhang2018learning,
          title={aprendendo uma única rede convolucional de super-resolução para múltiplas degradações},
          booktitle={ieee conference on computer vision and pattern recognition},
          pages={3262--3271},
          year={2018}
        }
    '''

    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap') # 'nearest' | 'mirror'
    x = bicubic_degradation(x, sf=sf)

    return x


def dpsr_degradation(x, k, sf=3):
    '''
    subamostragem bicúbica + borrão
    args:
        x: imagem hxwxc, [0, 1]
        k: hxw, dobro
        sf: fator de escala de subamostragem
    retorna:
        imagem lr com subamostragem bicúbica
    referência:
        @inproceedings{zhang2019deep,
          title={super-resolução profunda plug-and-play para kernels de desfoque arbitrários},
          booktitle={ieee conference on computer vision and pattern recognition},
          pages={1671--1681},
          year={2019}
        }
    '''

    x = bicubic_degradation(x, sf=sf)
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    
    return x


def classical_degradation(x, k, sf=3):
    '''desfoque + downsampling
    args:
        x: imagem hxwxc, [0, 1]/[0, 255]
        k: hxw, dobro
        sf: fator de escala inferior
    retorna:
        imagem lr com resolução reduzida
    '''

    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')

    # x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0

    return x[st::sf, st::sf, ...]


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """afiação usm. emprestado de real-esrgan
    imagem de entrada: i; imagem desfocada: b.
    1. k = i + peso * (i - b)
    2. máscara = 1 caso abs(i - b) > limite, ou então: 0
    3. máscara de desfoque:
    4. out = máscara * k + (1 - máscara) * i
    args:
        img (Numpy array): imagem de entrada, hwc, bgr; float32, [0, 1].
        weight (float): peso acentuado. padrão: 1.
        radius (float): tamanho do kernel do gaussian blur. padrão: 50.
        limite (int):
    """

    if radius % 2 == 0:
        radius += 1

    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur

    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')

    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)

    return soft_mask * K + (1 - soft_mask) * img


def add_blur(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf

    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()

        k = anisotropic_Gaussian(ksize=2 * random.randint(2, 11) + 3, theta=random.random() * np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2 * random.randint(2, 11) + 3, wd * random.random())

    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def add_resize(img, sf=4):
    rnum = np.random.rand()

    if rnum > 0.8: # cima
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7: # baixo
        sf1 = random.uniform(0.5 / sf, 1)
    else:
        sf1 = 1.0

    img = cv2.resize(img, (int(sf1 * img.shape[1]), int(sf1 * img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


# def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
#     noise_level = random.randint(noise_level1, noise_level2)
#
#     rnum = np.random.rand()
#
#     if rnum > 0.6:
#         img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
#     elif rnum < 0.4:
#         img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
#     else: # adição de ruído
#         L = noise_level2 / 255.
#         D = np.diag(np.random.rand(3))
#         U = orth(np.random.rand(3, 3))
#
#         conv = np.dot(np.dot(np.transpose(U), D), U)
#
#         img += np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
#
#     img = np.clip(img, 0.0, 1.0)
#
#     return img

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()

    if rnum > 0.6: # adiciona cor ruído gaussiano
        img = img + np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # adiciona ruído gaussiano em escala de cinza
        img = img + np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else: # adiciona ruído
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))

        conv = np.dot(np.dot(np.transpose(U), D), U)
        
        img = img + np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
    
    img = np.clip(img, 0.0, 1.0)

    return img


def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()

    if rnum > 0.6:
        img += img * np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img * np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))

        conv = np.dot(np.dot(np.transpose(U), D), U)

        img += img * np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
    
    img = np.clip(img, 0.0, 1.0)

    return img


def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10 ** (2 * random.random() + 2.0) # [2, 4]

    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.

        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        
        img += noise_gray[:, :, np.newaxis]

    img = np.clip(img, 0.0, 1.0)

    return img


def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)

    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)

    return img


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    h, w = lq.shape[:2]

    rnd_h = random.randint(0, h - lq_patchsize)
    rnd_w = random.randint(0, w - lq_patchsize)

    lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf, rnd_w_H:rnd_w_H + lq_patchsize * sf, :]
    
    return lq, hq


def degradation_bsrgan(img, sf=4, lq_patchsize=72, isp_model=None):
    """
    esse é o modelo de degradação de BSRGAN do artigo
    "designing a practical degradation model for deep blind image super-resolution"
    ----------
    img: hxwxc, [0, 1], seu tamanho deve ser maior que (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: fator de escala
    isp_model: modelo de isp da câmera
    retorna:    
    -------
    img: patch de baixa qualidade, tamanho: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: patch de alta qualidade, tamanho: (lq_patchsizexsf)X(lq_patchsizexsf)xc, range: [0, 1]
    """

    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
    sf_ori = sf

    h1, w1 = img.shape[:2]
    img = img.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...] # corte mod
    h, w = img.shape[:2]

    if h < lq_patchsize * sf or w < lq_patchsize * sf:
        raise ValueError(f'tamanho da imagem ({h1}X{w1}) é muito pequeno...')

    hq = img.copy()

    if sf == 4 and random.random() < scale2_prob: # downsample1
        if np.random.rand() < 0.5:
            img = cv2.resize(img, (int(1 / 2 * img.shape[1]), int(1 / 2 * img.shape[0])),
                             interpolation=random.choice([1, 2, 3]))
        else:
            img = util.imresize_np(img, 1 / 2, True)

        img = np.clip(img, 0.0, 1.0)

        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)

    if idx1 > idx2: # mantenha downsample3 por último
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)

        elif i == 1:
            img = add_blur(img, sf=sf)

        elif i == 2:
            a, b = img.shape[1], img.shape[0]

            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)

                img = cv2.resize(img, (int(1 / sf1 * img.shape[1]), int(1 / sf1 * img.shape[0])),
                                 interpolation=random.choice([1, 2, 3]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum() # blur com kernel deslocado

                img = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...] # subamostragem mais próxima

            img = np.clip(img, 0.0, 1.0)

        elif i == 3:
            # downsample3
            img = cv2.resize(img, (int(1 / sf * a), int(1 / sf * b)), interpolation=random.choice([1, 2, 3]))
            img = np.clip(img, 0.0, 1.0)

        elif i == 4:
            # adição de ruído gaussiano
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)

        elif i == 5:
            # adição de ruído jpeg
            if random.random() < jpeg_prob:
                img = add_JPEG_noise(img)

        elif i == 6:
            # adição de ruído do sensor da câmera processado
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)

    # adição de ruído jpeg final
    img = add_JPEG_noise(img)

    # corte aleatório
    img, hq = random_crop(img, hq, sf_ori, lq_patchsize)

    return img, hq


def degradation_bsrgan_variant(image, sf=4, isp_model=None):
    """
    esse é o modelo de degradação de BSRGAN do artigo
    "designing a practical degradation model for deep blind image super-resolution"
    ----------
    sf: fator de escala
    isp_model: modelo de isp da câmera
    retorna:
    -------
    img: patch de baixa qualidade, tamanho: lq_patchsizeXlq_patchsizeXC, intervalo: [0, 1]
    hq: patch de alta qualidade, tamanho: (lq_patchsizexsf)X(lq_patchsizexsf)xc, intervalo: [0, 1]
    """

    image = util.uint2single(image)
    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
    sf_ori = sf

    h1, w1 = image.shape[:2]
    image = image.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]
    h, w = image.shape[:2]

    hq = image.copy()

    if sf == 4 and random.random() < scale2_prob: # downsample1
        if np.random.rand() < 0.5:
            image = cv2.resize(image, (int(1 / 2 * image.shape[1]), int(1 / 2 * image.shape[0])),
                               interpolation=random.choice([1, 2, 3]))
        else:
            image = util.imresize_np(image, 1 / 2, True)
        image = np.clip(image, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)

    if idx1 > idx2: # mantém o downsample3 por último
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:
        if i == 0:
            image = add_blur(image, sf=sf)

        elif i == 1:
            image = add_blur(image, sf=sf)

        elif i == 2:
            a, b = image.shape[1], image.shape[0]

            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)
                image = cv2.resize(image, (int(1 / sf1 * image.shape[1]), int(1 / sf1 * image.shape[0])),
                                   interpolation=random.choice([1, 2, 3]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum() # desfoque com kernel deslocado

                image = ndimage.filters.convolve(image, np.expand_dims(k_shifted, axis=2), mode='mirror')
                image = image[0::sf, 0::sf, ...] # downsampling mais próximo

            image = np.clip(image, 0.0, 1.0)

        elif i == 3:
            # downsample3
            image = cv2.resize(image, (int(1 / sf * a), int(1 / sf * b)), interpolation=random.choice([1, 2, 3]))
            image = np.clip(image, 0.0, 1.0)

        elif i == 4:
            # adição de ruído gaussiano
            image = add_Gaussian_noise(image, noise_level1=2, noise_level2=25)

        elif i == 5:
            # adição de ruído jpeg
            if random.random() < jpeg_prob:
                image = add_JPEG_noise(image)

        # elif i == 6:
        #     # adição de ruído do sensor da câmera processado
        #     if random.random() < isp_prob and isp_model is not None:
        #         with torch.no_grad():
        #             img, hq = isp_model.forward(img.copy(), hq)

    # adição de ruído de compressão jpeg final
    image = add_JPEG_noise(image)
    image = util.single2uint(image)
    example = {"image":image}

    return example


def degradation_bsrgan_plus(img, sf=4, shuffle_prob=0.5, use_sharp=True, lq_patchsize=64, isp_model=None):
    """
    esse é um modelo de degradação estendido combinando
    os modelos de degradação de bsrgan e real-esrgan
    ----------
    img: hxwxc, [0, 1], seu tamanho deve ser maior que (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: fator de escala
    use_shuffle: o embaralhamento da degradação
    use_sharp: afiando a imagem
    retorna
    -------
    img: patch de baixa qualidade, tamanho: lq_patchsizeXlq_patchsizeXC, intervalo: [0, 1]
    hq: patch de alta qualidade correspondente, tamanho: (lq_patchsizexsf)X(lq_patchsizexsf)XC, intervalo: [0, 1]
    """

    h1, w1 = img.shape[:2]
    img = img.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]
    h, w = img.shape[:2]

    if h < lq_patchsize * sf or w < lq_patchsize * sf:
        raise ValueError(f'tamanho da imagem ({h1}X{w1}) é muito pequeno...')

    if use_sharp:
        img = add_sharpening(img)

    hq = img.copy()

    if random.random() < shuffle_prob:
        shuffle_order = random.sample(range(13), 13)
    else:
        shuffle_order = list(range(13))

        # embaralhamento local para ruído, jpeg é sempre o último
        shuffle_order[2:6] = random.sample(shuffle_order[2:6], len(range(2, 6)))
        shuffle_order[9:13] = random.sample(shuffle_order[9:13], len(range(9, 13)))

    poisson_prob, speckle_prob, isp_prob = 0.1, 0.1, 0.1

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)
        elif i == 1:
            img = add_resize(img, sf=sf)
        elif i == 2:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
        elif i == 3:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 4:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 5:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        elif i == 6:
            img = add_JPEG_noise(img)
        elif i == 7:
            img = add_blur(img, sf=sf)
        elif i == 8:
            img = add_resize(img, sf=sf)
        elif i == 9:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
        elif i == 10:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 11:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 12:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        else:
            print('verifica o shuffle!')

    # redimensiona para o tamanho desejado
    img = cv2.resize(img, (int(1 / sf * hq.shape[1]), int(1 / sf * hq.shape[0])),
                     interpolation=random.choice([1, 2, 3]))

    # adição de ruído de compressão jpeg final
    img = add_JPEG_noise(img)

    # corte aleatório
    img, hq = random_crop(img, hq, sf, lq_patchsize)

    return img, hq


if __name__ == '__main__':
	print("hey")
	img = util.imread_uint('utils/test.png', 3)

	print(img)
	img = util.uint2single(img)

	print(img)
	img = img[:448, :448]

	h = img.shape[0] // 4
	print("resizing to", h)

	sf = 4

	deg_fn = partial(degradation_bsrgan_variant, sf=sf)

	for i in range(20):
		print(i)

		img_lq = deg_fn(img)

		print(img_lq)

		img_lq_bicubic = albumentations.SmallestMaxSize(max_size=h, interpolation=cv2.INTER_CUBIC)(image=img)["image"]
		
        print(img_lq.shape)
        print("bicubic", img_lq_bicubic.shape)
		print(img_hq.shape)

		lq_nearest = cv2.resize(util.single2uint(img_lq), (int(sf * img_lq.shape[1]), int(sf * img_lq.shape[0])),
		                        interpolation=0)
    
		lq_bicubic_nearest = cv2.resize(util.single2uint(img_lq_bicubic), (int(sf * img_lq.shape[1]), int(sf * img_lq.shape[0])),
		                        interpolation=0)
    
		img_concat = np.concatenate([lq_bicubic_nearest, lq_nearest, util.single2uint(img_hq)], axis=1)
		
        util.imsave(img_concat, str(i) + '.png')