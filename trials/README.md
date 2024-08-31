# Trials
- `report_trials.json` is the given trial file, originally named `eval_object_categories.json`
- `resized_object_0.json` is DIY trials, using resized images.

### DIY Trials:
This directory contains the trials that were created by `generate_trials.py` script.
- We randomly sampled 5 trials per-image
- Each trial have 1 target and 3 foil images from other classes
- Using original images.

The trials are stored in the following format:
- `object_{seed}.json` where `seed` is the random seed used to generate the trials
- `fig_3b_{seed}.png` selected reported example pictures from paper's **figure 3b**, randomly generated trials on them.

