import numpy as np
import math
import glob
import os
import argparse
from statsmodels.stats.multitest import multipletests
import json
import sys
from collections import defaultdict
from tqdm import tqdm

from scipy.stats import ks_2samp, wasserstein_distance
from scipy.special import kl_div


def load_agg(f):

    if "json" in f:
        import json

        with open(f, "r") as fp:
            return json.load(fp)
    else:
        import pickle

        with open(f, "rb") as fp:
            return pickle.load(fp)


def run_comparison(target, reference, args, output_file_name=None):

    if os.path.isdir(args.output) and output_file_name is None:
        outfile = os.path.join(args.output, f"{os.path.basename(target)[:-4]}.json")
    else:
        outfile = output_file_name

    dist1 = load_agg(reference)
    dist2 = load_agg(target)

    if args.bin_phonemes is not None:
        phoneme_lookup = load_agg(args.bin_phonemes)
    else:
        phoneme_lookup = None

    results = {}
    features = set(dist1.keys()).intersection(set(dist2.keys()))

    for feature in features:

        if feature not in results:
            results[feature] = {}

        phn_data1 = dist1[feature]
        phn_data2 = dist2[feature]

        valid_phns1 = set([p for p, v in phn_data1.items() if len(v) > args.min_data])
        valid_phns2 = set([p for p, v in phn_data2.items() if len(v) > args.min_data])

        valid_phns = valid_phns1.intersection(valid_phns2)

        if len(valid_phns) <= 0:
            continue

        if args.mode == "ks":

            pvalues = [
                ks_2samp(phn_data1[phn], phn_data2[phn]).pvalue for phn in valid_phns
            ]

            if args.correction:
                pvalues = multipletests(pvalues, alpha=0.01, method="bonferroni")[1]

            for phn, pvalue in zip(valid_phns, pvalues):

                results[feature][phn] = float(pvalue)

        elif args.mode == "kl":

            raise NotImplementedError("KL Divergence not currently implemented")

            divergence = [kl_div(phn_data1[phn], phn_data2[phn]) for phn in valid_phns]

            for phn, div_value in zip(valid_phns, divergence):
                results[feature][phn] = float(div_value)

        elif args.mode == "wasserstein":

            pvalues = [
                ks_2samp(phn_data1[phn], phn_data2[phn]).pvalue for phn in valid_phns
            ]

            if args.correction:
                reject = multipletests(pvalues, alpha=0.01, method="bonferroni")[0]
            else:
                reject = [p < 0.01 for p in pvalues]

            distance = [
                wasserstein_distance(phn_data1[phn], phn_data2[phn]) if r else 0.0
                for phn, r in zip(valid_phns, reject)
            ]

            for phn, value in zip(valid_phns, distance):
                results[feature][phn] = float(value)

    if phoneme_lookup is not None:
        # Bin phonemes into classes and average the distance metric
        binned_results = {}
        for feature, phn_data in results.items():
            binned_results[feature] = defaultdict(list)
            for phn, value in phn_data.items():
                try:
                    # Phoneme already in lookup
                    phn_class = phoneme_lookup[phn]
                except KeyError:
                    phn_class = input(f"{phn} not found, class? ").strip().capitalize()
                    phoneme_lookup[phn] = phn_class

                binned_results[feature][phn_class].append(value)
            binned_results[feature] = {
                k: np.mean(v) for k, v in binned_results[feature].items()
            }
        results = binned_results

    # Save Updated Phoneme Lookup
    if phoneme_lookup is not None:
        with open(args.bin_phonemes, "w") as fp:
            json.dump(phoneme_lookup, fp)

    with open(outfile, "w") as f:
        json.dump(results, f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        "-r",
        help="Baseline distribution to compare against, if not given, performs full cross comparison",
        required=False,
    )
    parser.add_argument("aggregate2", nargs="+")
    parser.add_argument("--output", "-o", required=True)

    parser.add_argument("--min_data", type=int, default=0)

    parser.add_argument("--mode", choices=["ks", "kl", "wasserstein"], default="ks")
    parser.add_argument(
        "--correction",
        action="store_true",
        default=False,
        help="Apply multiple tests correction for each phoneme",
    )

    parser.add_argument(
        "--bin_phonemes",
        default=None,
        help="Path to phoneme lookup dictionary for phoneme classes",
    )

    parser.add_argument("--exclude_pattern", "-e", default=None)

    args = parser.parse_args()

    if args.reference:

        for agg2 in args.aggregate2:

            if agg2 == args.reference:
                continue
            if args.exclude_pattern is not None and args.exclude_pattern in agg2:
                continue
            else:
                run_comparison(agg2, args.reference, args)

    else:
        # Cross Comparison Mode
        with tqdm(total=math.comb(len(args.aggregate2), 2)) as pbar:
            for i, agg1 in enumerate(args.aggregate2[:-1]):
                for agg2 in args.aggregate2[i + 1 :]:
                    pbar.update(1)
                    if args.exclude_pattern is not None and (
                        args.exclude_pattern in agg2 or args.exclude_pattern in agg1
                    ):
                        continue

                    t1 = os.path.basename(agg1)[:-4]
                    t2 = os.path.basename(agg2)[:-4]
                    output_file_name = os.path.join(args.output, f"{t1}__{t2}.json")
                    run_comparison(agg1, agg2, args, output_file_name=output_file_name)


#         if os.path.isdir(args.output):
#             outfile = os.path.join(args.output, f"{os.path.basename(agg2)[:-4]}.json")

#         dist1 = load_agg(args.reference)
#         dist2 = load_agg(agg2)

#         if args.bin_phonemes is not None:
#             phoneme_lookup = load_agg(args.bin_phonemes)
#         else:
#             phoneme_lookup = None

#         results = {}
#         features = set(dist1.keys()).intersection(set(dist2.keys()))

#         for feature in features:

#             if feature not in results:
#                 results[feature] = {}

#             phn_data1 = dist1[feature]
#             phn_data2 = dist2[feature]

#             valid_phns1 = set(
#                 [p for p, v in phn_data1.items() if len(v) > args.min_data]
#             )
#             valid_phns2 = set(
#                 [p for p, v in phn_data2.items() if len(v) > args.min_data]
#             )

#             valid_phns = valid_phns1.intersection(valid_phns2)

#             if len(valid_phns) <= 0:
#                 continue

#             if args.mode == "ks":

#                 pvalues = [
#                     ks_2samp(phn_data1[phn], phn_data2[phn]).pvalue
#                     for phn in valid_phns
#                 ]

#                 if args.correction:
#                     pvalues = multipletests(pvalues, alpha=0.01, method="bonferroni")[1]

#                 for phn, pvalue in zip(valid_phns, pvalues):

#                     results[feature][phn] = float(pvalue)

#             elif args.mode == "kl":

#                 raise NotImplementedError("KL Divergence not currently implemented")

#                 divergence = [
#                     kl_div(phn_data1[phn], phn_data2[phn]) for phn in valid_phns
#                 ]

#                 for phn, div_value in zip(valid_phns, divergence):
#                     results[feature][phn] = float(div_value)

#             elif args.mode == "wasserstein":

#                 pvalues = [
#                     ks_2samp(phn_data1[phn], phn_data2[phn]).pvalue
#                     for phn in valid_phns
#                 ]

#                 if args.correction:
#                     reject = multipletests(pvalues, alpha=0.01, method="bonferroni")[0]
#                 else:
#                     reject = [p < 0.01 for p in pvalues]

#                 distance = [
#                     wasserstein_distance(phn_data1[phn], phn_data2[phn]) if r else 0.0
#                     for phn, r in zip(valid_phns, reject)
#                 ]

#                 for phn, value in zip(valid_phns, distance):
#                     results[feature][phn] = float(value)

#         if phoneme_lookup is not None:
#             # Bin phonemes into classes and average the distance metric
#             binned_results = {}
#             for feature, phn_data in results.items():
#                 binned_results[feature] = defaultdict(list)
#                 for phn, value in phn_data.items():
#                     try:
#                         # Phoneme already in lookup
#                         phn_class = phoneme_lookup[phn]
#                     except KeyError:
#                         phn_class = (
#                             input(f"{phn} not found, class? ").strip().capitalize()
#                         )
#                         phoneme_lookup[phn] = phn_class

#                     binned_results[feature][phn_class].append(value)
#                 binned_results[feature] = {
#                     k: np.mean(v) for k, v in binned_results[feature].items()
#                 }
#             results = binned_results

#         # Save Updated Phoneme Lookup
#         if phoneme_lookup is not None:
#             with open(args.bin_phonemes, "w") as fp:
#                 json.dump(phoneme_lookup, fp)

#         with open(outfile, "w") as f:
#             json.dump(results, f)


if __name__ == "__main__":
    main()
