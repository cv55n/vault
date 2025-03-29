import torch
from torch import nn


class DDPGradientStatsHook:
    def __init__(self, ddp_module):
        try:
            ddp_module.register_comm_hook(self, self._hook_fn)
        except AttributeError:
            raise ValueError('ddpgradientstatshook não suporta módulos não encapsulados em ddp')
        
        self._clear_state()

    def _clear_state(self):
        self.bucket_sq_norms_small_batch = []
        self.bucket_sq_norms_large_batch = []

    @staticmethod
    def _hook_fn(self, bucket):
        buf = bucket.buffer()

        self.bucket_sq_norms_small_batch.append(buf.pow(2).sum())

        fut = torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.AVG, async_op=True).get_future()
        
        def callback(fut):
            buf = fut.value()[0]

            self.bucket_sq_norms_large_batch.append(buf.pow(2).sum())

            return buf
        
        return fut.then(callback)

    def get_stats(self):
        sq_norm_small_batch = sum(self.bucket_sq_norms_small_batch)
        sq_norm_large_batch = sum(self.bucket_sq_norms_large_batch)

        self._clear_state()

        stats = torch.stack([sq_norm_small_batch, sq_norm_large_batch])

        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.AVG)

        return stats[0].item(), stats[1].item()


class GradientNoiseScale:
    """calcula a escala de ruído de gradiente (1 / snr), ou tamanho crítico do lote, de
    um modelo empírico de treinamento em lotes grandes.

    https://arxiv.org/abs/1812.06162.

    args:
        beta (float): o fator de decaimento para as médias móveis exponenciais usadas
            para calcular a escala de ruído de gradiente.
            padrão: 0.9998

        eps (float): adicionado para estabilidade numérica.
            padrão: 1e-8
    """

    def __init__(self, beta=0.9998, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.ema_sq_norm = 0.
        self.ema_var = 0.
        self.beta_cumprod = 1.
        self.gradient_noise_scale = float('nan')

    def state_dict(self):
        """retorna o estado do objeto como um :class:`dict`."""

        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """carrega o estado do objeto.

        args:
            state_dict (dict): estado do objeto. deve ser um objeto retornado de uma
            chamada para :meth:`state_dict`.
        """

        self.__dict__.update(state_dict)

    def update(self, sq_norm_small_batch, sq_norm_large_batch, n_small_batch, n_large_batch):
        """atualiza o estado com as estatísticas de gradiente de um novo lote e retorna a
        escala de ruído de gradiente atual.

        args:
            sq_norm_small_batch (float): a média das normas 2 quadradas dos gradientes
                de microlote ou por amostra.

            sq_norm_large_batch (float): a norma 2 quadrada da média do microlote ou
                gradientes por amostra.

            n_small_batch (int): o tamanho do lote do microlote individual ou por amostra
                gradientes (1 caso por sample).

            n_large_batch (int): o tamanho total do lote da média do microlote ou
                por gradientes de amostra.
        """

        est_sq_norm = (n_large_batch * sq_norm_large_batch - n_small_batch * sq_norm_small_batch) / (n_large_batch - n_small_batch)
        est_var = (sq_norm_small_batch - sq_norm_large_batch) / (1 / n_small_batch - 1 / n_large_batch)
        
        self.ema_sq_norm = self.beta * self.ema_sq_norm + (1 - self.beta) * est_sq_norm
        self.ema_var = self.beta * self.ema_var + (1 - self.beta) * est_var
        self.beta_cumprod *= self.beta
        self.gradient_noise_scale = max(self.ema_var, self.eps) / max(self.ema_sq_norm, self.eps)
        
        return self.gradient_noise_scale

    def get_gns(self):
        """retorna a escala de ruído do gradiente atual."""

        return self.gradient_noise_scale

    def get_stats(self):
        """retorna as estimativas atuais (desviadas) do gradiente médio quadrado
        e da variância do gradiente."""

        return self.ema_sq_norm / (1 - self.beta_cumprod), self.ema_var / (1 - self.beta_cumprod)