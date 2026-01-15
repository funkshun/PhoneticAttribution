import librosa as lb
import numpy as np
from tqdm.auto import tqdm
from bisect import bisect_left
from sklearn.linear_model import LinearRegression
import textgrid

import argparse
import os
import pickle
from collections import defaultdict
import math
import glob
import pathlib
import sys

from scipy.signal import correlate
import parselmouth
from parselmouth.praat import call
from parselmouth import AmplitudeScaling
import feature_extraction as fe


# Handling Phonemes (Loading TextGrid and Creating Lookup)
class PhonemeMapping:

    def __init__(self, phns):

        self.phns = []
        self.starts = []
        self.end = -1

        self.lookup = defaultdict(list)

        for phn, start, end in phns:
            self.phns.append(phn)
            self.starts.append(start)
            if end > self.end:
                self.end = end

            self.lookup[phn].append((start, end))

    def get_phn(self, time):
        return self.phns[bisect_left(self.starts, time) - 1]

    def get_times(self, phn):
        return self.lookup[phn]


def load_phones(f, strip_silence=False):

    if f.find(".TextGrid") < 0:
        sf = glob.glob(f"{f[:-4]}*.TextGrid")
        if len(sf) <= 0:
            raise FileNotFoundError(f"Could not find TextGrid for file {f}")
        else:
            f = sf[0]

    tg = textgrid.TextGrid.fromFile(f)

    # Apparently there is inconsistency in which TextGrid level to load, so determine live
    tg_tier = np.argmin([np.mean([len(i.mark) for i in tier]) for tier in tg])

    results = []
    for i in tg[tg_tier]:
        if strip_silence and (i.mark == "(...)" or i.mark == ""):
            continue
        results.append((i.mark, i.minTime, i.maxTime))

    return PhonemeMapping(results)


def get_time_series(l, fl, fh, center=True):

    start = 0.0 if center else fl / 2
    end = l if center else l - (fl / 2)
    return np.arange(start, end, step=fh)


def get_class_fname_split(f, class_idx):

    parts = f.split("/")
    class_name = parts[-class_idx]
    file_name = "/".join(parts[-class_idx:])

    return class_name, file_name


def extract_f0(y, sr, frame_length=0.06, frame_hop=0.03, fmin=65, fmax=2093):

    fl = int(sr * frame_length)
    fh = int(sr * frame_hop)

    f0, voicing, voiceP = lb.pyin(
        y=y, sr=sr, fmin=fmin, fmax=fmax, frame_length=fl, hop_length=fh
    )

    times = get_time_series(len(y) / sr, frame_length, frame_hop, center=True)

    return f0, voiceP, times


# Feature Extraction Methods
def extract_zcr(y, sr, frame_length=0.06, frame_hop=0.03):

    fl = int(sr * frame_length)
    fh = int(sr * frame_hop)

    l = len(y) / sr
    ts = get_time_series(l, frame_length, frame_hop, center=True)

    return lb.feature.zero_crossing_rate(y, frame_length=fl, hop_length=fh)[0], ts


def extract_hnr(
    sound,
    harmonics_type="preferred",
    time_step=0.01,
    min_time=0.0,
    max_time=0.0,
    minimum_pitch=75.0,
    silence_threshold=0.1,
    num_periods_per_window=4.5,
    interpolation_method="Parabolic",
    return_values=False,
    replacement_for_nan=0.0,
):

    # Create a Harmonicity object
    if harmonics_type == "preferred":
        harmonicity = call(
            sound,
            "To Harmonicity (cc)",
            time_step,
            minimum_pitch,
            silence_threshold,
            num_periods_per_window,
        )
    elif harmonics_type == "ac":
        harmonicity = call(
            sound,
            "To Harmonicity (ac)",
            time_step,
            minimum_pitch,
            silence_threshold,
            num_periods_per_window,
        )
    else:
        return (None, None)

    harmonicity_values = [
        call(harmonicity, "Get value in frame", frame_no)
        for frame_no in range(len(harmonicity))
    ]
    times = [
        call(harmonicity, "Get time from frame number", frame_no + 1)
        for frame_no in range(len(harmonicity))
    ]
    # Convert NaN values to floats (default: 0)
    harmonicity_values = [
        value if not math.isnan(value) else replacement_for_nan
        for value in harmonicity_values
    ]

    return harmonicity_values, times


def extract_formant_ratios(sound):

    time, formants = fe.get_n_formants(sound)

    R1 = {
        "feature": [
            f2 / f1 if (f1 != 0.0 and f2 != 0.0) else np.nan
            for f1, f2 in zip(formants[0], formants[1])
        ],
        "times": time,
    }
    R2 = {
        "feature": [
            f2 / f1 if (f1 != 0.0 and f2 != 0.0) else np.nan
            for f1, f2 in zip(formants[1], formants[2])
        ],
        "times": time,
    }
    return {"R1": R1, "R2": R2}


def extract_spectral(y, sr, frame_length=0.06, frame_hop=0.03, n_fft=2048):

    fl = int(sr * frame_length)
    fh = int(sr * frame_hop)

    times = get_time_series(len(y) / sr, frame_length, frame_hop)
    S, _ = lb.magphase(lb.stft(y=y, n_fft=n_fft, hop_length=fh, win_length=fl))

    res = {}

    res["intensity"] = {"feature": lb.feature.rms(S=S)[0], "times": times}
    res["spec_centroid"] = {
        "feature": lb.feature.spectral_centroid(S=S)[0],
        "times": times,
    }
    res["spec_bandwidth"] = {
        "feature": lb.feature.spectral_bandwidth(S=S)[0],
        "times": times,
    }
    # Spectral Tilt through Least Squares Regression
    bins = np.array(list(range(S.shape[0])))
    tilt = np.array([calc_tilt(S[:, t_indx], bins) for t_indx in range(S.shape[-1])])
    res["spec_tilt"] = {"feature": tilt, "times": times}

    return res


def calc_tilt(frame, bins):

    X = bins.reshape(-1, 1)
    # y = frame.reshape(-1, 1)

    reg = LinearRegression().fit(X=X, y=frame)
    return reg.coef_[0]


def file2key(f):

    return f"{os.path.basename(f)[:-4]}.pkl"


def pload(f):

    with open(f, "rb") as fp:
        return pickle.load(fp)


# For each file, load phns, run feature extractors and save
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--id_list", help="List of basename patterns to run")
    parser.add_argument("--separate_classes", type=int, default=-1)

    # Feature Flags
    parser.add_argument("--zcr", action="store_true", default=False)
    parser.add_argument("--hnr", action="store_true", default=False)
    parser.add_argument("--f0", action="store_true", default=False)
    parser.add_argument("--spectral", action="store_true", default=False)
    parser.add_argument("--formant", action="store_true", default=False)

    # Handle Existing Files, default is to skip existing files
    existing_args = parser.add_mutually_exclusive_group()
    existing_args.add_argument("--rerun", action="store_true", default=False)
    existing_args.add_argument("--update", action="store_true", default=False)
    existing_args.add_argument("--replace", action="store_true", default=False)

    parser.add_argument("--update_phonemes", action="store_true", default=False)

    # Audio Options
    parser.add_argument("--sr", type=int, default=16000)

    args = parser.parse_args()
    if args.rerun:
        args.update = True

    if not args.rerun and not os.path.isdir(args.output):
        os.mkdir(args.output)

    if len(args.files) == 1 and os.path.isdir(args.files[0]):
        args.files = glob.glob(os.path.join(args.files[0], "**/*.wav"), recursive=True)
    elif all([os.path.isdir(inf) for inf in args.files]):
        temp_files = []
        for inf in args.files:
            temp_files.extend(glob.glob(os.path.join(inf, "**/*.wav"), recursive=True))
        args.files = temp_files

    print(f"## Found {len(args.files)} Initial Files ##")

    classes, fnames = None, None
    if args.separate_classes >= 0:
        joined_results = [
            get_class_fname_split(f, args.separate_classes) for f in args.files
        ]
        classes = set([v[0] for v in joined_results])
        fnames = [v[1] for v in joined_results]

        for c in classes:
            class_dir = os.path.join(args.output, c)
            if not os.path.isdir(class_dir):
                os.mkdir(class_dir)

    if classes is not None:
        print(f"## Files Separated into {len(classes)} Classes ##")
        print(classes)

    if args.id_list:
        with open(args.id_list, "r") as f:
            chosen_files = set([s.strip() for s in f.readlines()])
    else:
        chosen_files = None

    if not args.rerun and fnames is not None:
        args.files = [
            (f, of)
            for f, of in zip(args.files, fnames)
            if (
                chosen_files is None
                or os.path.basename(f) in chosen_files
                or os.path.basename(f)[:10] in chosen_files
            )  # LJspeech ID list
        ]
    elif not args.rerun:
        args.files = [
            f
            for f in args.files
            if (
                chosen_files is None  # No file filtering
                or os.path.basename(f) in chosen_files  # Direct basename list
                or os.path.basename(f)[:10] in chosen_files  # LJspeech ID list
            )
        ]

    print(f"## Final Files: {len(args.files)} ##")

    for i, fn in tqdm(enumerate(args.files), total=len(args.files)):

        if args.rerun:
            data = pload(fn)
            f = data["source"]
            outfile = fn
        else:
            if fnames is not None:
                f, suffix = fn
                outfile = os.path.join(args.output, f"{suffix[:-4]}.pkl")
            else:
                f = fn
                outfile = os.path.join(args.output, file2key(f))

            if os.path.exists(outfile) and not (args.update or args.replace):
                continue

            if os.path.exists(outfile) and args.update:
                data = pload(outfile)
            else:
                data = {}
            if not os.path.exists(os.path.dirname(outfile)):
                pathlib.Path(os.path.dirname(outfile)).mkdir(
                    parents=True, exist_ok=True
                )

            abspath = os.path.abspath(f)
            data["source"] = abspath

        if "phonemes" not in data or args.update_phonemes:
            try:
                data["phonemes"] = load_phones(f)
            except FileNotFoundError:
                continue

        y, sr = lb.load(f, sr=args.sr)

        if args.zcr:
            feature, time_series = extract_zcr(y, sr)
            data["zcr"] = {"feature": feature, "times": time_series}

        if args.f0:
            f0, voiceP, time_series = extract_f0(y, sr)
            data["f0"] = {"feature": f0, "probability": voiceP, "times": time_series}

        if args.spectral:
            spectral_features = extract_spectral(y, sr)
            data.update(spectral_features)

        # Parselmouth Features
        pm_sound = parselmouth.Sound(f)
        if args.hnr:
            hnr, times = extract_hnr(pm_sound, minimum_pitch=65)
            data["hnr"] = {"feature": hnr, "times": times}
        if args.formant:
            formant_features = extract_formant_ratios(pm_sound)
            data.update(formant_features)

        with open(outfile, "wb") as fp:
            pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
