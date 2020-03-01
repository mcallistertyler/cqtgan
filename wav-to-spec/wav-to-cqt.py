## Turns wav files into CQT Spectrogram for use in some Image-to-Image Translation model
from nnAudio import Spectrogram
from scipy.io import wavfile
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sr, track = wavfile.read('16-012.wav')
x = torch.tensor(track).float()
x = x.cuda()
p = (84 - 256) // 2
print('x shape', x.shape)
x = F.pad(x, (p, p), "reflect").squeeze(1)
print('x shape', x)
spec_layer = Spectrogram.CQT1992v2(sr=sr, device='cpu', output_format='Magnitude')
spec = spec_layer(x)
spec = torch.log(torch.clamp(spec, min=1e-5))
plt.imsave('test2.png', spec[0].cpu().numpy())
