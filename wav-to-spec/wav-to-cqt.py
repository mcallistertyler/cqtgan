## Turns wav files into CQT Spectrogram for use in some Image-to-Image Translation model
from spec2wav.dataset import AudioDataset
from torch.utils.data import DataLoader
import argparse
from nnAudio import Spectrogram
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

def save_spec_images(spec, name):
    imgdt = datetime.datetime.now()
    plt.imsave(str(name) + '.png', spec[0].cpu().numpy())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_type", type=str, default="torch")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    save_type = args.save_type
    spec_layer = Spectrogram.CQT1992v2(sr=22050, n_bins=84, hop_length=256, pad_mode='constant', device='cuda:0', verbose=False, trainable=False, output_format='Magnitude')
    transformedSet = AudioDataset('input_audio.txt', 22050 * 4, sampling_rate=22050, augment=False)
    transformedLoader = DataLoader(transformedSet, batch_size=1)
    transformedVoc = []
    f = open('input_audio.txt', 'r')
    lines = f.readlines()
    lines = list(map(lambda s: s.strip(), lines)) #remove newline character
    lines = [track.replace('.wav', '') for track in lines] #remove .wav
    print(lines)
    if len(lines) != len(transformedLoader):
        print('Differences in wavs found and whats in input_audio.txt')
        return

    for i, x_t in enumerate(transformedLoader):
        x_t = x_t.cuda()
        s_t = spec_layer(x_t).detach()
        s_t = torch.log(torch.clamp(s_t, min=1e-5))
        transformedVoc.append(s_t.cuda())
    
    if (save_type == 'torch'):
        print('Saving WAVs as torch pt')
        for x in range(0, len(transformedVoc)):
            torch.save(transformedVoc[x], lines[x] + '.pt')
    if (save_type == 'png'):
        print('Saving WAVs as image via matplotlib')
        for x in range(0, len(transformedVoc)):
            save_spec_images(transformedVoc[x], lines[x])

if __name__ == "__main__":
    main()