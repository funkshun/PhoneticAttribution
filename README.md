# May I Ask Who’s Speaking? Phonetic Source Attribution of Synthetic Speech

This repository contains the code and selected samples from the datasets in the referenced paper.

The `source_attribution` directory contains the main implementation of our attribution approach.
Model definitions are in the `microdetector.py` and `fusion_module.py` files.
Training and testing scripts have cli interfaces with usage guidelines.

The `phonetic_analysis` directory contains feature extraction and analysis code needed to genereate phonetic comparisons.
Each script has a cli interface with usage guidelines.

The `dataset_samples` directory includes example audio files from each generator to allow for relative quality inspections.

Full datasets and model checkpoints will be released at the time of publication.


