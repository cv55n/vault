"""utilidades para o monodepth."""

import sys
import re
import numpy as np
import cv2
import torch


def read_pfm(path):
    """lê o arquivo pfm.

    args:
        path (str): caminho para arquivo

    retorna:
        tuple: (data, scale)
    """

    with open(path, "rb") as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()

        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("não é um arquivo pfm: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))

        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("cabeçalho pfm malformado.")

        scale = float(file.readline().decode("ascii").rstrip())

        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """escreve um arquivo pfm.

    args:
        path (str): caminho para o arquivo
        image (array): dados
        scale (int, opcional): escala. o padrão é 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("o tipo de imagem deve ser float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3: # imagem da cor
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ): # escala de cinza
            color = False
        else:
            raise Exception("a imagem precisa ter as dimensões H x W x 3, H x W x 1 ou H x W.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def read_image(path):
    """lê a imagem e gera uma imagem rgb (0-1).

    args:
        path (str): caminho para o arquivo

    retorna:
        array: imagem rgb (0-1)
    """

    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def resize_image(img):
    """redimensiona a imagem e a torna adequada para a rede.

    args:
        img (array): imagem

    retorna:
        tensor: dados prontos para a rede
    """

    height_orig = img.shape[0]
    width_orig = img.shape[1]

    if width_orig > height_orig:
        scale = width_orig / 384
    else:
        scale = height_orig / 384

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )

    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """redimensiona o mapa de profundidade e traz para o cpu (numpy).

    args:
        depth (tensor): profundidade
        width (int): largura da imagem
        height (int): altura da imagem

    retorna:
        array: profundidade processada
    """

    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")

    depth_resized = cv2.resize(
        depth.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )

    return depth_resized

def write_depth(path, depth, bits=1):
    """escreve o mapa de profundidade em um arquivo pfm e png.

    args:
        path (str): caminho para o arquivo sem extensão
        depth (array): profundidade
    """
    
    write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return