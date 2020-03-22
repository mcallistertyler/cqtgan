## Turns wav files into CQT Spectrogram for use in some Image-to-Image Translation model
## Find average normalization values...I guess
from mel2wav.dataset import AudioDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
import argparse
from PIL import Image
from nnAudio import Spectrogram
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

min_A = []
max_A = []

def inverse_normalisation(min, max):
    im = Image.open('original_3_fake_A.png').convert('L')
    im = ToTensor()(im)
    # tim = torch.load('torch/generated_0.pt')
    # print('image', im)
    # print('image shaep', im.shape)
    # print('loaded tensor', tim)
    denormalised_spec = im[0] * (max - min) + min 
    print('denormalised', denormalised_spec)
    print('saving denormalised')
    torch.save(denormalised_spec, 'denormalised-03-fake-A.pt')
    
def track_normalisation_average(voc):
    for spec in voc:
        for i in enumerate(spec):
            min_A.append(torch.min(spec[0]))
            max_A.append(torch.max(spec[0]))
            print('min a', min_A)
            print('max a', max_A)
            inverse_normalisation(-8.7912, 2.9027)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_type", type=str, default="torch")
    parser.add_argument("--inverse_normal", type=bool, default=False)
    parser.add_argument("--input_file", type=str, default="input.txt")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    save_type = args.save_type
    file_name = args.input_file
    spec_layer = Spectrogram.CQT1992v2(sr=22050, n_bins=84, hop_length=256, pad_mode='constant', device='cuda:0', verbose=False, trainable=False, output_format='Magnitude')
    transformedSet = AudioDataset(file_name, 22050 * 4, sampling_rate=22050, augment=False)
    transformedLoader = DataLoader(transformedSet, batch_size=1)
    transformedVoc = []
    f = open(file_name, 'r')
    lines = f.readlines()
    lines = list(map(lambda s: s.strip(), lines)) #remove newline character
    lines = [track.replace('.wav', '') for track in lines] #remove .wav
    lines = [track.split("/")[-1] for track in lines]
    if len(lines) != len(transformedLoader):
        print('Differences in wavs found and whats in input_audio.txt')
        return

    for i, x_t in tqdm(enumerate(transformedLoader), ascii=True, desc='Making spectrogram representations'):
        x_t = x_t.cuda()
        s_t = spec_layer(x_t).detach()
        s_t = torch.log(torch.clamp(s_t, min=1e-5))
        transformedVoc.append(s_t.cuda())
        track_normalisation_average(transformedVoc)
    
    if (save_type == 'none'):
        print('Only running inversion')
    if (save_type == 'torch'):
        for x in tqdm(range(0, len(transformedVoc)), ascii=True, desc='Making torch tensors'):
            torch.save(transformedVoc[x], 'torch/' + lines[x] + '.pt')
    if (save_type == 'png'):
        for x in tqdm(range(0, len(transformedVoc)), ascii=True, desc='Making pngs'):
            save_image(transformedVoc[x], 'png/' + lines[x] + '.png', normalize=True)

if __name__ == "__main__":
    main()
