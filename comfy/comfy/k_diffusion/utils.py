from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import shutil
import urllib
import warnings

from PIL import Image
import torch
from torch import nn, optim
from torch.utils import data
from torchvision.transforms import functional as TF


def from_pil_image(x):
    """converte uma imagem pil em um tensor."""

    x = TF.to_tensor(x)

    if x.ndim == 2:
        x = x[..., None]

    return x * 2 - 1


def to_pil_image(x):
    """converte um tensor em uma imagem pil."""

    if x.ndim == 4:
        assert x.shape[0] == 1

        x = x[0]

    if x.shape[0] == 1:
        x = x[0]

    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)


def hf_datasets_augs_helper(examples, transform, image_key, mode='RGB'):
    """aplicar transformações passadas para conjuntos de dados huggingface."""

    images = [transform(image.convert(mode)) for image in examples[image_key]]

    return {image_key: images}


def append_dims(x, target_dims):
    """acrescenta dimensões ao final de um tensor até que ele tenha dimensões target_dims."""
    
    dims_to_append = target_dims - x.ndim

    if dims_to_append < 0:
        raise ValueError(f'a entrada tem {x.ndim} dims mas target_dims é {target_dims}, o que é menos')
    
    expanded = x[(...,) + (None,) * dims_to_append]

    # o mps obterá valores inf se tentar indexar nos novos eixos, mas desanexar corrige isso.
    # https://github.com/pytorch/pytorch/issues/84364

    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded


def n_params(module):
    """retorna o número de parâmetros treináveis em um módulo."""

    return sum(p.numel() for p in module.parameters())


def download_file(path, url, digest=None):
    """baixa um arquivo se ele não existir, opcionalmente verificando seu hash sha-256."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not path.exists():
        with urllib.request.urlopen(url) as response, open(path, 'wb') as f:
            shutil.copyfileobj(response, f)

    if digest is not None:
        file_digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()

        if digest != file_digest:
            raise OSError(f'hash de {path} (url: {url}) não foi possível validar')
        
    return path


@contextmanager
def train_mode(model, mode=True):
    """um gerenciador de contexto que coloca um modelo no modo de treinamento e restaura
    o modo anterior na saída."""

    modes = [module.training for module in model.modules()]

    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """um gerenciador de contexto que coloca um modelo no modo de avaliação e restaura
    o modo anterior na saída."""
    
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """incorpora parâmetros de modelo atualizados em uma versão de média móvel exponencial
    de um modelo. deve ser chamado após cada etapa do otimizador."""

    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())

    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())

    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class EMAWarmup:
    """implementa um aquecimento ema usando um cronograma de decaimento inverso.
    se inv_gamma=1 e power=1, implementa uma média simples. inv_gamma=1, power=2/3 são
    bons valores para modelos que você planeja treinar para um milhão ou mais passos (atinge fator de decaimento
    0,999 em 31,6 mil passos, 0,9999 em 1 milhão de passos), inv_gamma=1, power=3/4 para modelos
    que você planeja treinar para menos (atinge fator de decaimento 0,999 em 10 mil passos, 0,9999 em
    215,4 mil passos).
    args:
        inv_gamma (float): fator multiplicativo inverso do aquecimento da ema. padrão: 1.
        power (float): fator exponencial do aquecimento da ema. padrão: 1.
        min_value (float): a taxa mínima de decaimento da ema. padrão: 0.
        max_value (float): a taxa máxima de decaimento da ema. padrão: 1.
        start_at (int): a época para começar a média. padrão: 0.
        last_epoch (int): o índice da última época. padrão: 0.
    """

    def __init__(self, inv_gamma=1., power=1., min_value=0., max_value=1., start_at=0,
                 last_epoch=0):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self):
        """retorna o estado da classe como um :class:`dict`."""

        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """carrega o estado da classe.
        args:
            state_dict (dict): estado do scaler. deve ser um objeto retornado
                de uma chamada para :meth:`state_dict`.
        """

        self.__dict__.update(state_dict)

    def get_value(self):
        """obtém a taxa de decaimento atual da ema."""

        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        return 0. if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """atualiza a contagem de passos."""

        self.last_epoch += 1


class InverseLR(optim.lr_scheduler._LRScheduler):
    """implementa um cronograma de taxa de aprendizado de decaimento inverso com um
    aquecimento exponencial opcional. quando last_epoch=-1, define lr inicial como lr.
    inv_gamma é o número de etapas/épocas necessárias para que a taxa de aprendizagem decai para
    (1 / 2)**power of its original value.
    args:
        optimizer (otimizador): otimizador encapsulado.
        inv_gamma (float): fator multiplicativo inverso de decaimento da taxa de aprendizagem. padrão: 1.
        power (float): fator exponencial de decaimento da taxa de aprendizado. padrão: 1.
        warmup (float): fator de aquecimento exponencial (0 <= aquecimento < 1, 0 para desabilitar). padrão: 0.
        min_lr (float): a taxa mínima de aprendizado. padrão: 0.
        last_epoch (int): o índice da última época. padrão: -1.
        verbose (bool): se ``true``, imprime uma mensagem para stdout para cada atualização. padrão: ``false``.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., min_lr=0.,
                 last_epoch=-1, verbose=False):
        self.inv_gamma = inv_gamma
        self.power = power

        if not 0. <= warmup < 1:
            raise ValueError('valor inválido para aquecimento')
        
        self.warmup = warmup
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("para obter a última taxa de aprendizagem calculada pelo planejador, "
                          "por favor utilize `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)

        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power

        return [warmup * max(self.min_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class ExponentialLR(optim.lr_scheduler._LRScheduler):
    """implementa um cronograma de taxa de aprendizado exponencial com um aquecimento exponencial
    opcional. quando last_epoch=-1, define lr inicial como lr. decai a taxa de aprendizado
    continuamente por decaimento (padrão 0,5) a cada num_steps passos.
    args:
        optimizer (otimizador): otimizador encapsulado.
        num_steps (float): o número de etapas para decair a taxa de aprendizagem por decaimento.
        decay (float): o fator pelo qual a taxa de aprendizagem deve decair a cada passos num_steps. padrão: 0.5.
        warmup (float): fator de aquecimento exponencial (0 <= aquecimento < 1, 0 para desabilitar). padrão: 0.
        min_lr (float): a taxa mínima de aprendizado. padrão: 0.
        last_epoch (int): o índice da última época. padrão: -1.
        verbose (bool): se ``true``, imprime uma mensagem no stdout para cada atualização. padrão: ``false``.
    """

    def __init__(self, optimizer, num_steps, decay=0.5, warmup=0., min_lr=0.,
                 last_epoch=-1, verbose=False):
        self.num_steps = num_steps
        self.decay = decay

        if not 0. <= warmup < 1:
            raise ValueError('valor inválido para aquecimento')
        
        self.warmup = warmup
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("para obter a última taxa de aprendizado calculada pelo planejador, "
                          "use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)

        lr_mult = (self.decay ** (1 / self.num_steps)) ** self.last_epoch

        return [warmup * max(self.min_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """extrai amostras de uma distribuição lognormal."""

    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()


def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """extrai amostras de uma distribuição log-logística opcionalmente truncada."""

    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)

    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()

    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """extrai amostras de uma distribuição logarítmica uniforme."""

    min_value = math.log(min_value)
    max_value = math.log(max_value)

    return (torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """extrai amostras de uma distribuição de passo de tempo de treinamento de difusão v truncada."""
    
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi

    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf

    return torch.tan(u * math.pi / 2) * sigma_data


def rand_split_log_normal(shape, loc, scale_1, scale_2, device='cpu', dtype=torch.float32):
    """extrai amostras de uma distribuição lognormal dividida."""

    n = torch.randn(shape, device=device, dtype=dtype).abs()
    u = torch.rand(shape, device=device, dtype=dtype)

    n_left = n * -scale_1 + loc
    n_right = n * scale_2 + loc

    ratio = scale_1 / (scale_1 + scale_2)

    return torch.where(u < ratio, n_left, n_right).exp()


class FolderOfImages(data.Dataset):
    """encontra recursivamente todas as imagens em um diretório. não suporta
    classes/alvos."""

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

    def __init__(self, root, transform=None):
        super().__init__()

        self.root = Path(root)
        self.transform = nn.Identity() if transform is None else transform
        self.paths = sorted(path for path in self.root.rglob('*') if path.suffix.lower() in self.IMG_EXTENSIONS)

    def __repr__(self):
        return f'FolderOfImages(root="{self.root}", len: {len(self)})'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        path = self.paths[key]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        image = self.transform(image)

        return image,


class CSVLogger:
    def __init__(self, filename, columns):
        self.filename = Path(filename)
        self.columns = columns

        if self.filename.exists():
            self.file = open(self.filename, 'a')
        else:
            self.file = open(self.filename, 'w')
            self.write(*self.columns)

    def write(self, *args):
        print(*args, sep=',', file=self.file, flush=True)


@contextmanager
def tf32_mode(cudnn=None, matmul=None):
    """um gerenciador de contexto que define se o tf32 é permitido no cudnn ou no matmul."""

    cudnn_old = torch.backends.cudnn.allow_tf32
    matmul_old = torch.backends.cuda.matmul.allow_tf32

    try:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul
        yield
    finally:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_old
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul_old