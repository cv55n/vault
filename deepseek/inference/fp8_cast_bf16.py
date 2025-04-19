import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant


def main(fp8_path, bf16_path):
    """
    converte pesos fp8 em bf16 e salva os pesos convertidos.

    essa função lê pesos fp8 do diretório especificado, converte-os para bf16,
    e salva os pesos convertidos em outro diretório especificado. ela também atualiza o
    arquivo de índice do modelo para refletir as alterações.

    args:
    fp8_path (str): o caminho para o diretório que contém os pesos do fp8 e o arquivo de índice do modelo.
    bf16_path (str): o caminho para o diretório onde os pesos bf16 convertidos serão salvos.

    gera:
    keyerror: se um tensor scale_inv necessário estiver faltando para um peso.

    notas:
    - a função assume que os pesos do FP8 são armazenados em arquivos safetensor.
    - a função armazena em cache os arquivos safetensor carregados para otimizar o uso da memória.
    - a função atualiza o arquivo de índice do modelo para remover referências aos tensores scale_inv.
    """

    torch.set_default_dtype(torch.bfloat16)

    os.makedirs(bf16_path, exist_ok=True)

    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)

    weight_map = model_index["weight_map"]

    # cache para arquivos safetensor carregados
    loaded_files = {}
    fp8_weight_names = []

    # função auxiliar para obter tensor do arquivo correto
    def get_tensor(tensor_name):
        """
        recupera um tensor dos arquivos safetensor armazenados em cache ou o carrega do disco, caso não esteja armazenado em cache.
        
        args:
            tensor_name (str): o nome do tensor a ser recuperado.

        retorna:
            torch.Tensor: o tensor recuperado.

        gera:
            keyerror: se o tensor não existir no arquivo safetensor.
        """

        file_name = weight_map[tensor_name]

        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")

        return loaded_files[file_name][tensor_name]
    
    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}

        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1: # peso fp8
                scale_inv_name = f"{weight_name}_scale_inv"

                try:
                    # obtém scale_inv do arquivo correto
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)

                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"aviso: faltando tensor scale_inv para [x], ignorando conversão")
                    
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # gerenciamento de memória: mantenha apenas os 2 arquivos usados ​​mais recentemente
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))

            del loaded_files[oldest_file]

            torch.cuda.empty_cache()
    
    # atualiza o índice do modelo
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")

    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"

        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
            
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)