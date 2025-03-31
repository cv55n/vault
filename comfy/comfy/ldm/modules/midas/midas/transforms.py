import numpy as np
import cv2
import math


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """redimensiona a amostra para garantir o tamanho fornecido. mantém a proporção de aspecto.

    args:
        sample (dict): amostra
        size (tuple): tamanho da imagem

    retorna:
        tuple: novo tamanho
    """

    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # redimensiona
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )

    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST
    )

    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """
    redimensiona a amostra para o tamanho fornecido (largura, altura).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA
    ):
        """inicializa.

        args:
            width (int): largura desejada
            height (int): altura desejada
            resize_target (bool, opcional):
                True: redimensiona a amostra inteira (imagem, máscara, alvo).
                False: redimensiona apenas a imagem.
                padrão é true.
            keep_aspect_ratio (bool, opcional):
                True: manter a proporção de aspecto da amostra de entrada.
                a amostra de saída pode não ter a largura e altura fornecidas, e
                o comportamento de redimensionamento depende do parâmetro 'resize_method'.
                padrão é False.
            ensure_multiple_of (int, opcional):
                a largura e altura de saída são restritas a ser um múltiplo deste parâmetro.
                padrão é 1.
            resize_method (str, opcional):
                "lower_bound": a saída será pelo menos tão grande quanto o tamanho fornecido.
                "upper_bound": a saída será no máximo tão grande quanto o tamanho fornecido. (o tamanho da saída pode ser menor que o tamanho fornecido.)
                "minimal": escalar o mínimo possível. (o tamanho da saída pode ser menor que o tamanho fornecido.)
                padrão é "lower_bound".
        """

        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determina a nova altura e largura
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # escala tal que o tamanho da saída é o limite inferior
                if scale_width > scale_height:
                    # ajusta a largura
                    scale_height = scale_width
                else:
                    # ajusta a altura
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # escala tal que o tamanho da saída é o limite superior
                if scale_width < scale_height:
                    # ajustar largura
                    scale_height = scale_width
                else:
                    # ajusta a altura
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # escala o mínimo possível
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # ajusta a largura
                    scale_height = scale_width
                else:
                    # ajusta a altura
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} não implementado"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )

            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} não implementado")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # redimensiona a amostra
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )
            
            sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """
    normaliza a imagem pela média e desvio padrão fornecidos.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """
    prepara a amostra para uso como entrada da rede.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample