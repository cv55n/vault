import torch
from torch import nn


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('a decadência deve estar entre 0 e 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates
        
        else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remover já que o caractere '.' não é permitido em buffers
                s_name = name.replace('.', '')

                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates

        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1

            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())

        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        salva os parâmetros atuais para restaurar mais tarde.
        args:
          parameters: iterável de `torch.nn.parameter`; os parâmetros a serem
            armazenados temporariamente.
        """

        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        restaura os parâmetros armazenados com o método `store`.
        útil para validar o modelo com parâmetros ema sem afetar o
        processo de otimização original. armazena os parâmetros antes do
        método `copy_to`. após a validação (ou salvamento do modelo), use isso para
        restaurar os parâmetros anteriores.
        args:
          parameters: iterável de `torch.nn.parameter`; os parâmetros a serem
            atualizados com os parâmetros armazenados.
        """

        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)