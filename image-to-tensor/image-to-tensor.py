from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ToTensor, Normalize, ToPILImage
from torchvision.utils import save_image
import torch.nn.functional as f
from matplotlib import pyplot
from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

image = Image.open('1000-1.png').convert('L')
print('image', image)
image = ToTensor()(image)
print('image to tensor', image)
loaded = torch.load('1000-1.pt')
print('loaded tensor', loaded)
ski = imread('1000-1.png') 
print('scikit image', ski)
ski = rgb2gray(ski)
print('scikti image to gray', ski)
ski = ToTensor()(ski)
print('scikit to tensor', ski)
mat = pyplot.imread('1000-1.png')
print('matplotlib', mat)
mat = ToTensor()(mat)
print('mat plot lit to tensor', mat)
x = loaded
min_ele = torch.min(x)
x -= min_ele
x /= torch.max(x)
#print(x)