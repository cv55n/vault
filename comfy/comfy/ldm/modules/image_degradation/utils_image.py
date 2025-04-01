import os
import math
import random
import numpy as np
import torch
import cv2
from torchvision.utils import make_grid
from datetime import datetime

# todo: checar com o dominik
# import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


'''
# --------------------------------------------
# https://github.com/twhui/SRGAN-pyTorch
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')

    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def surf(Z, cmap='rainbow', figsize=None):
    plt.figure(figsize=figsize)
    ax3 = plt.axes(projection='3d')

    w, h = Z.shape[:2]
    xx = np.arange(0,w,1)
    yy = np.arange(0,h,1)
    X, Y = np.meshgrid(xx, yy)

    ax3.plot_surface(X,Y,Z,cmap=cmap)
    # ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap=cmap)

    plt.show()


'''
# --------------------------------------------
# obtém os caminhos de imagem
# --------------------------------------------
'''

def get_image_paths(dataroot):
    paths = None # retorna none se dataroot for none

    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))

    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} não é um diretório válido'.format(path)

    images = []

    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)

    assert images, '{:s} não tem arquivo de imagem válido'.format(path)

    return images


'''
# --------------------------------------------
# divide imagens grandes em imagens pequenas
# --------------------------------------------
'''

def patches_from_image(img, p_size=512, p_overlap=64, p_max=800):
    w, h = img.shape[:2]

    patches = []

    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=np.int))

        w1.append(w-p_size)
        h1.append(h-p_size)
        
        # print(w1)
        # print(h1)
        
        for i in w1:
            for j in h1:
                patches.append(img[i:i+p_size, j:j+p_size,:])
    else:
        patches.append(img)

    return patches


def imssave(imgs, img_path):
    """
    imgs: lista, x imagens de tamanho wxhxc
    """

    img_name, ext = os.path.splitext(os.path.basename(img_path))

    for i, img in enumerate(imgs):
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]

        new_path = os.path.join(os.path.dirname(img_path), img_name+str('_s{:04d}'.format(i))+'.png')
        
        cv2.imwrite(new_path, img)


def split_imageset(original_dataroot, taget_dataroot, n_channels=3, p_size=800, p_overlap=96, p_max=1000):
    """
    divida as imagens grandes de original_dataroot em pequenas imagens sobrepostas com tamanho (p_size)x(p_size),
    e salve-as em taget_dataroot; somente as imagens com tamanho maior que (p_max)x(p_max)
    serão divididas.
    args:
        original_dataroot:
        taget_dataroot:
        p_size: tamanho das imagens pequenas
        p_overlap: tamanho do patch no treinamento é uma boa escolha
        p_max: imagens com tamanho menor que (p_max)x(p_max) manter inalterado.
    """

    paths = get_image_paths(original_dataroot)

    for img_path in paths:
        # img_name, ext = os.path.splitext(os.path.basename(img_path))
        img = imread_uint(img_path, n_channels=n_channels)

        patches = patches_from_image(img, p_size, p_overlap, p_max)
        imssave(patches, os.path.join(taget_dataroot,os.path.basename(img_path)))

        # if original_dataroot == taget_dataroot:
        # del img_path

'''
# --------------------------------------------
# makedir
# --------------------------------------------
'''

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()

        print('o caminho já existe. renomeie-o para [{:s}]'.format(new_name))

        os.rename(path, new_name)

    os.makedirs(path)


'''
# --------------------------------------------
# lê a imagem do caminho
# opencv é rápido, mas lê imagem numpy bgr
# --------------------------------------------
'''

# --------------------------------------------
# obtém a imagem uint8 de tamanho hxwxn_channels (rgb)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    # input: path

    # saída: hxwx3(rgb ou ggg), ou hxwx1 (g)
    if n_channels == 1:
        img = cv2.imread(path, 0) # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2) # hxwx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # bgr ou g

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # ggg
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # rgb

    return img


# --------------------------------------------
# imwrite do matlab
# --------------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)

    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]

    cv2.imwrite(img_path, img)

def imwrite(img, img_path):
    img = np.squeeze(img)

    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]

    cv2.imwrite(img_path, img)


# --------------------------------------------
# obtém a imagem única de tamanho hxwxn_channels (bgr)
# --------------------------------------------
def read_img(path):
    # lê a imagem por cv2
    # return: float32 numpy, hwc, bgr, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    # algumas imagens tem 4 canais
    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


'''
# --------------------------------------------
# conversão de formato de imagem
# --------------------------------------------
# numpy(single) <--->  numpy(unit)
# numpy(single) <--->  tensor
# numpy(unit)   <--->  tensor
# --------------------------------------------
'''

# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(unit)
# --------------------------------------------

def uint2single(img):
    return np.float32(img/255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())


def uint162single(img):
    return np.float32(img/65535.)


def single2uint16(img):
    return np.uint16((img.clip(0, 1)*65535.).round())


# --------------------------------------------
# numpy(unit) (hxwxc ou hxw) <--->  tensor
# --------------------------------------------

# converte um uint para tensor de tocha de 4 dimensões
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)


# converte um uint para tensor de tocha tridimensional
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


# converte um tensor de tocha 2/3/4-dimensional para uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return np.uint8((img*255.0).round())


# --------------------------------------------
# numpy(single) (HxWxC) <--->  tensor
# --------------------------------------------

# converte um tensor de tocha simples (axlxc) em tridimensional
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# converte um tensor de tocha simples (axlxc) em tensor de tocha quadridimensional
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


# converte um tensor do torch para simples
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img

# converte um tensor do torch para simples
def tensor2single3(img):
    img = img.data.squeeze().float().cpu().numpy()

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    return img


def single2tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float().unsqueeze(0)


def single32tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)


def single42tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float()


# de skimage.io importar imread, imsave
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    converte um tensor de tocha em uma matriz numpy de imagem de ordem de canal bgr
    input: 4d(b,(3/1),h,w), 3d(c,h,w), ou 2d(h,w), qualquer intervalo, ordem dos canais rgb
    output: 3d(h,w,c) ou 2d(h,w), [0,255], np.uint8 (padrão)
    '''

    tensor = tensor.squeeze().float().cpu().clamp_(*min_max) # aperta primeiro e depois prende
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0]) # variação [0,1]
    
    n_dim = tensor.dim()

    if n_dim == 4:
        n_img = len(tensor)

        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)) # hwc, bgr
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)) # hwc, bgr
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'suporta apenas tensor 4d, 3d e 2d. mas recebido com dimensão: {:d}'.format(n_dim))
    
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

        # importante. ao contrário do matlab, numpy.unit8() não arredondará por padrão.

    return img_np.astype(out_type)


'''
# --------------------------------------------
# aumento, inverte e/ou gira
# --------------------------------------------
# os dois seguintes são suficientes.
# (1) augmet_img: imagem numpy de wxhxc ou wxh
# (2) augment_img_tensor4: imagem tensor 1xcxwxh
# --------------------------------------------
'''

def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def augment_img_tensor4(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])


def augment_img_tensor(img, mode=0):
    img_size = img.size()
    img_np = img.data.cpu().numpy()

    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))

    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))

    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)


def augment_img_np3(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return img.transpose(1, 0, 2)
    elif mode == 2:
        return img[::-1, :, :]
    elif mode == 3:
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)

        return img
    elif mode == 4:
        return img[:, ::-1, :]
    elif mode == 5:
        img = img[:, ::-1, :]
        img = img.transpose(1, 0, 2)

        return img
    elif mode == 6:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]

        return img
    elif mode == 7:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)

        return img


def augment_imgs(img_list, hflip=True, rot=True):
    # vira horizontalmente ou gira
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)

        return img

    return [_augment(img) for img in img_list]


'''
# --------------------------------------------
# modcrop e barbear
# --------------------------------------------
'''

def modcrop(img_in, scale):
    # img_in: numpy, hwc ou hw
    img = np.copy(img_in)

    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale

        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale

        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('ndim da imagem errado: [{:d}].'.format(img.ndim))
    
    return img


def shave(img_in, border=0):
    # img_in: numpy, hwc ou hw
    img = np.copy(img_in)

    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]

    return img


'''
# --------------------------------------------
# processo de processamento de imagem em imagem numpy
# channel_convert(in_c, tar_type, img_list):
# rgb2ycbcr(img, only_y=True):
# bgr2ycbcr(img, only_y=True):
# ycbcr2rgb(img):
# --------------------------------------------
'''

def rgb2ycbcr(img, only_y=True):
    '''o mesmo que matlab rgb2ycbcr
    only_y: retorna apenas o canal y
    entrada:
        uint8, [0, 255]
        float, [0, 1]
    '''

    in_img_type = img.dtype
    img.astype(np.float32)
    
    if in_img_type != np.uint8:
        img *= 255.
    
    # converte
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
        
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''o mesmo que matlab ycbcr2rgb
    entrada:
        uint8, [0, 255]
        float, [0, 1]
    '''

    in_img_type = img.dtype
    img.astype(np.float32)
    
    if in_img_type != np.uint8:
        img *= 255.
    
    # converte
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''versão bgr de rgb2ycbcr
    only_y: retorna apenas o canal y
    entrada:
        uint8, [0, 255]
        float, [0, 1]
    '''

    in_img_type = img.dtype
    img.astype(np.float32)
    
    if in_img_type != np.uint8:
        img *= 255.
    
    # converte
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_img_type)


def channel_convert(in_c, tar_type, img_list):
    # conversão entre bgr, cinza e y
    if in_c == 3 and tar_type == 'gray': # bgr para cinza
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]

        return [np.expand_dims(img, axis=2) for img in gray_list]
    
    elif in_c == 3 and tar_type == 'y': # bgr para y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]

        return [np.expand_dims(img, axis=2) for img in y_list]
    
    elif in_c == 1 and tar_type == 'RGB': # cinza/y para bgr
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


'''
# --------------------------------------------
# métrica, psnr e ssim
# --------------------------------------------
'''

# --------------------------------------------
# psnr
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    # img1 e img2 tem alcance [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    
    if not img1.shape == img2.shape:
        raise ValueError('as imagens de entrada devem ter as mesmas dimensões.')
    
    h, w = img1.shape[:2]
    
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2)**2)

    if mse == 0:
        return float('inf')
    
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# ssim
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calcula o ssim
    as mesmas saídas do matlab
    img1, img2: [0, 255]
    '''

    # img1 = img1.squeeze()
    # img2 = img2.squeeze()

    if not img1.shape == img2.shape:
        raise ValueError('as imagens de entrada devem ter as mesmas dimensões.')
    
    h, w = img1.shape[:2]

    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []

            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))

            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('dimensões de imagem de entrada incorretas.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # válido
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


'''
# --------------------------------------------
# imresize bicúbico do matlab (numpy e torch) [0, 1]
# --------------------------------------------
'''

# função 'imresize' do matlab, agora suporta apenas 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3

    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # usa um kernel modificado para interpolar e suavizar simultaneamente - maior largura do kernel
        kernel_width = kernel_width / scale

    # coordenadas do espaço de saída
    x = torch.linspace(1, out_length, out_length)

    # coordenadas de espaço de entrada. calcula o mapeamento inverso de modo que 0,5
    # no espaço de saída mapeie para 0,5 no espaço de entrada, e 0,5 + escala no espaço de saída
    # mapeie para 1,5 no espaço de entrada
    u = x / scale + 0.5 * (1 - 1 / scale)

    # qual é o pixel mais à esquerda que pode estar envolvido no cálculo?
    left = torch.floor(u - kernel_width / 2)

    # qual é o número máximo de pixels que podem estar envolvidos na
    # computação? nota: não tem problema usar um pixel extra aqui; se os
    # pesos correspondentes forem todos zero, ele será eliminado no final
    # desta função
    P = math.ceil(kernel_width) + 2

    # os índices dos pixels de entrada envolvidos no cálculo do k-ésimo pixel de saída
    # estão na linha k da matriz de índices
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # os pesos usados ​​para calcular o k-ésimo pixel de saída estão na linha k da
    # matriz de pesos
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices

    # aplica o kernel cúbico
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    # normaliza a matriz de pesos para que cada linha some 1
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # se uma coluna em pesos for toda zero, livre-se dela. considera apenas a primeira e a última coluna.
    weights_zero_tmp = torch.sum((weights == 0), 0)

    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)

    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)

    weights = weights.contiguous()
    indices = indices.contiguous()

    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length

    indices = indices + sym_len_s - 1

    return weights, indices, int(sym_len_s), int(sym_len_e)


# --------------------------------------------
# imresize para imagem tensor [0, 1]
# --------------------------------------------
def imresize(img, scale, antialiasing=True):
    # agora a escala deve ser a mesma para h e w
    # input: img: tensor pytorch, chw ou hw [0,1]
    # output: chw ou hw [0,1] c/o round
    need_squeeze = True if img.dim() == 2 else False
    
    if need_squeeze:
        img.unsqueeze_(0)
    
    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    
    kernel_width = 4
    kernel = 'cubic'

    # retorna a ordem de dimensão desejada para executar o redimensionamento. a
    # estratégia é executar o redimensionamento primeiro ao longo da dimensão com o
    # menor fator de escala
    #
    # não há suporte para isso

    # obtém os pesos e os índices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    
    # dimensão h do processo
    # cópia simétrica
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # dimensão w do processo
    # cópia simétrica
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)

    for i in range(out_W):
        idx = int(indices_W[i][0])

        for j in range(out_C):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_W[i])

    if need_squeeze:
        out_2.squeeze_()

    return out_2


# --------------------------------------------
# imresize para imagem numpy [0, 1]
# --------------------------------------------
def imresize_np(img, scale, antialiasing=True):
    # agora a escala deve ser a mesma para h e w
    # input: img: numpy, hwc ou hw [0,1]
    # output: hwc ou hw [0,1] com/ou round
    img = torch.from_numpy(img)
    
    need_squeeze = True if img.dim() == 2 else False

    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # retorna a ordem de dimensão desejada para executar o redimensionamento. a estratégia
    # é realizar o redimensionamento primeiro ao longo da dimensão com o menor fator
    # de escala
    #
    # não há suporte para isso

    # obtém os pesos e os índices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    
    # dimensão h do processo
    # cópia simétrica
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)

    for i in range(out_H):
        idx = int(indices_H[i][0])

        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # dimensão w do processo
    # cópia simétrica
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)

    for i in range(out_W):
        idx = int(indices_W[i][0])

        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])

    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()


if __name__ == '__main__':
    print('---')

#     img = imread_uint('test.bmp', 3)
#     img = uint2single(img)
#     img_bicubic = imresize_np(img, 1/4)