import glob
import textgrid
from bisect import bisect_left
from collections import defaultdict
import numpy as np


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


def determine_phone_timing(phns):

    if all([s < 15 for s in phns.starts]):
        return "seconds"
    elif any([s > 1024 for s in phns.starts]):
        return "samples"
    else:
        return "frames"


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
