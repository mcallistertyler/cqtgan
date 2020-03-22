from mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--pt", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path)

    args.save_path.mkdir(exist_ok=True, parents=True)

    spectrogram = torch.load(args.pt)
    if (len(spectrogram.shape) == 2):
        spectrogram = spectrogram.unsqueeze(0)
    reconstruction = vocoder.inverse(spectrogram).squeeze().cpu().numpy()
    librosa.output.write_wav(args.save_path / 'output-fake-A-03', reconstruction, sr=22050)

if __name__ == "__main__":
    main()
