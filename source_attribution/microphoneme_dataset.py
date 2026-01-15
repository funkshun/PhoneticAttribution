import os
import glob
import json
import textgrid as tg
import librosa as lb
import numpy as np
from tqdm.auto import tqdm

import torch
from torchaudio.compliance.kaldi import fbank
from torch.utils.data import Dataset
from scipy.signal import resample
from scipy.interpolate import interp1d, Akima1DInterpolator
from torchvision.transforms import Resize, InterpolationMode
from torchvision.transforms.functional import to_grayscale


class MicrophonemeDataset(Dataset):

    def __init__(
        self, root, phoneme, metadata, classsize=None, transform=None, sr=16000
    ):

        self.dirs = list(glob.glob(os.path.join(root, "*/")))
        self.sr = 16000
        self.phoneme = phoneme
        self.classes = {}
        self.labels = set()
        self.segments = []
        self.transform = transform

        if not os.path.exists(metadata):
            raise FileNotFoundError(f"Failed to find {metadata}")
        with open(metadata, "r") as fp:

            obj = json.load(fp)

            self.labels = sorted(list(set([k for v in obj.values() for k in v])))
            self.classes = obj

        for d in tqdm(self.dirs):
            class_name = os.path.basename(os.path.normpath(d))

            if class_name not in self.classes:
                raise ValueError(f"Class '{class_name}' not in metadata")

            loc_files = glob.glob(os.path.join(d, "**/*.wav"), recursive=True)
            loc_files = [f for f in loc_files if os.path.exists(f"{f[:-4]}.TextGrid")]

            if classsize:
                loc_files = np.random.choice(
                    loc_files, size=min(len(loc_files), classsize), replace=False
                )

            for f in loc_files:
                self.segments.extend(self._getSections(f, class_name, self.phoneme))

    def __getitem__(self, index):

        file, cname, start, end = self.segments[index]

        y, _ = lb.load(file, sr=self.sr)
        sstart = int(self.sr * start)
        send = int(self.sr * end)

        if self.transform is not None and self.transform.requires_bounds:
            data = self.transform(y, sstart, send)
        elif self.transform is not None and not self.transform.requires_bounds:
            data = self.transform(y[sstart:send])
        else:
            data = y

        label = self._cname2vector(cname).to(torch.float)

        return data, label

    def __len__(self):
        return len(self.segments)

    def _getSections(self, file, class_name, phoneme):

        sections = []
        grid = self._loadTextGrid(file)
        if grid is None:
            return sections
        for intv in grid:
            if intv.mark != phoneme:
                continue
            sections.append((file, class_name, intv.minTime, intv.maxTime))
        return sections

    def _loadTextGrid(self, file):

        grid = tg.TextGrid.fromFile(f"{file[:-4]}.TextGrid")
        for tier in grid:
            if tier.name == "phones":
                return tier
        return None

    def _cname2vector(self, cname):

        obj = self.classes[cname]

        return torch.tensor([1 if l in obj else 0 for l in self.labels])

    def _parse_metadata(self, obj):

        labels = []
        values = []
        for k, v in obj.items():
            labels.append(k)
            values.append(v)

        return labels, values


class InterpolatedMelSpectrogram:

    def __init__(
        self,
        target_len,
        sr=16000,
        n_fft=2048,
        hop_length=512,
        mode=InterpolationMode.BILINEAR,
    ):

        self.requires_bounds = False

        self.target_len = target_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.resize = Resize(size=(128, self.target_len))

    def __call__(self, data):

        S = torch.tensor(
            lb.feature.melspectrogram(
                y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
        ).unsqueeze(0)
        Snew = self.resize(S).squeeze()
        S3Chan = torch.stack([Snew, Snew, Snew])

        return S3Chan


class CropInterpMelSpectrogram:

    def __init__(
        self,
        target_len,
        sr=16000,
        n_fft=2048,
        hop_length=512,
        mode=InterpolationMode.BILINEAR,
    ):

        self.requires_bounds = True
        self.target_len = target_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.resize = Resize(size=(128, self.target_len))

    def __call__(self, data, start, stop):
        S = lb.feature.melspectrogram(
            y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        sframe, eframe = lb.core.samples_to_frames(
            [start, stop], hop_length=self.hop_length
        )

        S = torch.tensor(S[:, sframe:eframe]).unsqueeze(0)
        Snew = self.resize(S).squeeze()
        S3Chan = torch.stack([Snew, Snew, Snew])

        return S3Chan


class InterpolatedMelFbank:

    def __init__(
        self,
        target_len,
        sr=16000,
        n_fft=2048,
        hop_length=512,
        mode=InterpolationMode.BILINEAR,
    ):

        self.requires_bounds = False

        self.target_len = target_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.resize = Resize(size=(128, self.target_len))

    def __call__(self, data):

        S = fbank(
            torch.tensor(data).unsqueeze(0),
            num_mel_bins=128,
            sample_frequency=self.sr,
            frame_shift=(self.hop_length / 16),
            frame_length=(self.n_fft / 16),
        ).unsqueeze(0)

        Snew = self.resize(S).squeeze()
        S3Chan = torch.stack([Snew, Snew, Snew])

        return S3Chan
