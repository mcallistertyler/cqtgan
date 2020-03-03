## Turns wav files into CQT Spectrogram for use in some Image-to-Image Translation model
from spec2wav.dataset import AudioDataset
from nnAudio import Spectrogram
from scipy.io import wavfile
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import librosa
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime

def save_spec_images(spec):
    imgdt = datetime.datetime.now()
    plt.imsave(str(imgdt) + '.png', spec[0].cpu().numpy())

spec_layer = Spectrogram.CQT1992v2(sr=22050, n_bins=84, hop_length=256, pad_mode='constant', device='cuda:0', verbose=False, trainable=False, output_format='Magnitude')
transformedSet = AudioDataset('test_files.txt', 22050 * 4, sampling_rate=22050, augment=False)
transformedLoader = DataLoader(transformedSet, batch_size=1)
transformedVoc = []
for i, x_t in enumerate(transformedLoader):
    x_t = x_t.cuda()
    s_t = spec_layer(x_t).detach()
    s_t = torch.log(torch.clamp(s_t, min=1e-5))
    transformedVoc.append(s_t.cuda())
save_spec_images(transformedVoc[0])
torch.save(transformedVoc[0].cpu(), 'unseen.pt')
# x = torch.tensor(track).float()
# x = x.cuda()
# #p = (84 - 256) // 2
# print(x.shape)
# print('x shape', x)
# spec = spec_layer(x)
# spec = torch.log(torch.clamp(spec, min=1e-5))
# #plt.imsave('test2.png', spec[0].cpu().numpy())
# print('voc shape', spec.shape)
# #np.save('nparray4', spec.cpu().numpy())
# torch.save(spec.cpu(), 'spec.pt')
# #print('voc numpy shape', spec.cpu().numpy().shape)
