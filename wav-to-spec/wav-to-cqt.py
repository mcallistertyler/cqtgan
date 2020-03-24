## Turns wav files into CQT Spectrogram
from mel2wav.dataset import AudioDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import json
from nnAudio import Spectrogram
import torch
from tqdm import tqdm

normalisation_dict = {}

def track_normalisation_average(voc, names):
    for idx, spec in enumerate(voc):
        min_val = torch.min(spec[0]).item()
        max_val = torch.max(spec[0]).item()
        normalisation_dict[names[idx]] = { "min": min_val, "max": max_val }
    with open('normalisation_values.json', 'w') as outfile:
        json.dump(normalisation_dict, outfile, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="input.txt")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
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
    track_normalisation_average(transformedVoc, lines)
    
    for x in tqdm(range(0, len(transformedVoc)), ascii=True, desc='Making pngs'):
        save_image(transformedVoc[x], 'png/' + lines[x] + '.png', normalize=True)

if __name__ == "__main__":
    main()
