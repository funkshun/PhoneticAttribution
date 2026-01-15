import numpy as np
import argparse
from collections import defaultdict
import glob
import os
import pickle
from tqdm.auto import tqdm
import sys


class PhonemeMapping:

    def __init__(self, phns):

        from collections import defaultdict

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
        from bisect import bisect_left

        return self.phns[bisect_left(self.starts, time) - 1]

    def get_times(self, phn):
        return self.lookup[phn]


def aggmean(xs):
    return np.mean(xs)


def aggmedian(xs):
    return np.median(xs)


def aggcontour(xs):

    slope = xs[-1] - xs[0]
    maxprop = np.argmax(xs) / len(xs)
    minprop = np.argmin(xs) / len(xs)

    return (slope, maxprop, minprop)


def aggident(xs):
    return xs


def load_file(f):
    with open(f, "rb") as fp:
        return pickle.load(fp)


def minmax_norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def aggregate_feature(feature, times, phones, normalize=False, nan_tolereance=0.5):

    nan_prop = sum([np.isnan(v) for v in feature]) / len(feature)
    if nan_prop > nan_tolereance:
        return [], []
    phn_assignments = [phones.get_phn(t) for t in times]

    if normalize:
        feature = minmax_norm(feature)

    feat_agg = []
    phn_agg = []

    cur_phn = None
    for f, p in zip(feature, phn_assignments):

        if cur_phn is None or p != cur_phn:
            cur_phn = p
            phn_agg.append(cur_phn)
            feat_agg.append([])

        feat_agg[-1].append(f)

    return feat_agg, phn_agg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="+",
        help="Input directories, all .pkl files in subdirectories will be included",
    )
    parser.add_argument("--output", "-o", required=True, help="Output file")

    parser.add_argument(
        "--aggf",
        choices=["none", "mean", "median", "contour"],
        required=True,
        default="none",
    )
    parser.add_argument("--nan_tolerance", type=float, default=0.5)

    args = parser.parse_args()

    aggregate = defaultdict(dict)

    aggf_lookup = {
        "mean": aggmean,
        "median": aggmedian,
        "contour": aggcontour,
        "none": aggident,
    }
    aggf = aggf_lookup[args.aggf]

    if len(args.input) > 1 and not os.path.isdir(args.output):
        print("# If providing multiple input directories, output must be a directory #")
        sys.exit(1)

    args.output = os.path.normpath(args.output)

    for input_dir in args.input:

        input_dir = os.path.normpath(input_dir)

        if not os.path.isdir(input_dir):
            print(f"# Input Directory {input_dir} Not Found, Skipping... #")
            continue

        if os.path.isdir(args.output):
            outfile = os.path.join(args.output, f"{os.path.basename(input_dir)}.pkl")
        else:
            outfile = args.output

        input_files = glob.glob(os.path.join(input_dir, "**/*.pkl"), recursive=True)

        for f in tqdm(input_files):

            data = load_file(f)
            features = [
                k for k in data.keys() if k.lower() not in ["source", "phonemes"]
            ]

            for feat in features:

                if feat not in aggregate:
                    aggregate[feat] = defaultdict(list)

                normalize = feat == "intensity"

                feature_data = data[feat]["feature"]
                feature_time = data[feat]["times"]
                feat_agg, phn_agg = aggregate_feature(
                    feature_data,
                    feature_time,
                    data["phonemes"],
                    normalize=normalize,
                    nan_tolereance=args.nan_tolerance,
                )

                for f, p in zip(feat_agg, phn_agg):
                    if aggf is not None:
                        aggregate[feat][p].append(aggf(f))
                    else:
                        aggregate[feat][p].append(f)

        with open(outfile, "wb") as fp:
            pickle.dump(aggregate, fp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
