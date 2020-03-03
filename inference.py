from spec2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch
import numpy as np
from scipy.io import wavfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--load_spec", type=Path, required=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path)
    spec = np.load('nparray4.npy')
    spec = torch.from_numpy(spec)
    recons = vocoder.inverse(spec)
    print(recons.dtype)
    recons = recons.cpu().numpy()
    print(recons.shape)
    wavfile.write("res.wav", 22050, (recons.reshape((-1))*2**15).astype(np.int16))
    #librosa.output.write_wav("generated_test2", recons, sr=25500)

if __name__ == "__main__":
    main()
