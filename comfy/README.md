# comfy

a powerful modular stable diffusion gui

# installing

git clone the repo

put your sd checkpoints (ckpt/safetensors files) in: models/checkpoints

put your vae in: models/vae

## amd

amd users can install rocm and pytorch with pip if you don't have it already installed:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2```

## nvidia

nvidia users should install xformers

### dependencies

install the dependencies:

```pip install -r requirements.txt```

# running

```python main.py```

# notes

only parts of the graph that have an output with all the correct inputs will be executed.
 
only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. if you change the last part of the graph only the part you changed and the part that depends on it will be executed.

### fedora

to get python 3.10 on fedora:

```dnf install python3.10```

then you can:

```python3.10 -m ensurepip```
