import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """carrega o modelo do arquivo.

        args:
            path (str): caminho para o arquivo
        """

        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)