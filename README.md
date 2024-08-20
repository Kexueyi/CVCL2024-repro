
# CVCL2024-Repro

![teaser](figs/intro.png)

This code tries to reproduce the 2024 Vong, W. K. et al. paper: [Grounded language acquisition through the eyes and ears of a single child](https://www.science.org/doi/abs/10.1126/science.adi1374). This study uses a CLIP-like Model trained on visual-linguistic data([SAYCam-S](https://pubmed.ncbi.nlm.nih.gov/34485795/)) from a single child to show how basic representation can be learned from infant's everyday sensory input.

## Prerequisites

#### Prepare dataset

Since we don't have the direct access to exact [SAYCam-S](https://pubmed.ncbi.nlm.nih.gov/34485795/) dataset that was used in the paper, but this paper provide generalization example on [KonkObject](http://olivalab.mit.edu/MM/archives/ObjectCategories.zip) dataset(["Massive Memory" Object Categories](https://konklab.fas.harvard.edu/#)), we follow their way to prepare the data and evaluate the model for zero-shot object recognition under a [special setting](####trial-zero-shot).

1. Download the [KonkObject](http://olivalab.mit.edu/MM/archives/ObjectCategories.zip) dataset and [Classes](http://olivalab.mit.edu/MM/downloads/MM2-Ranks.xls), move classes to your dataset folder.
2. Update paths in run.sh.

#### Trial Zero-shot: A Special Setting
In contrast traditional generalized zero-shot learning, this paper define zero-shot "trials" as follows:
1. Select 4 images as 1 trial. 1 image is the target image and the other 3 images are foil images.
2. Pass 4 images and 1 target label to the CVCL model.
3. Get max similarity score in this 4-choose-1 trial.

#### Generate Trials 
We generate trials for the KonkObject dataset as described in paper's supplementary material, after filtering using baby's vocabulary, producing 5 trials per image and 85 trials per category, 5440 trials in total. Details can be found in `generate_trials.py`.

## Run

To reproduce the get zero-shot-trial results, simply run the following command:

```bash
$ ./repro.sh
```
To reproduce from trial generation to final results, run the following command:

```bash
$ rm dataset/trials/*.json
$ ./repro.sh
```
Generated trials can be found in `dataset/trials` folder.
Results can be found in `results` folder.

#### Repro Class-wise Accuracy
After running `run.sh`(run on 5 seeds), use `repro_plt.ipynb` to reproduce the class-wise accuracy as shown in the original study (Fig.3.A).

In repro plt:
1. Get box plt of class-wise accuracy of CVCL for 5 seeds.
2. Get CLIP & CVCL's total accuracy by avg of 5 seeds for comparison.

#### Repro Genralization Example

For generalization examples (Fig.3.B), run the corresponding section in `repro_plt.ipynb`.

In repro plt:
1. Maually select examples appear in the paper, filter out the corresponding trials. 
2. Get each images zero-shot-trial results and avg for each img's trials to get results.

#### Repro Comparison 
Class-wise accuracy comparison:
![Comparison-Class Accuracy](figs/cls_acc_comp.png)
Generalization example score comparison:
![Comparison-Example Score](figs/gen_exp_comp.png)
All figures can be found under `figs`.

## Open questions
- Why repro class-wise accuracy is different from the paper? 
  - Especially the Order of classes. (e.g. 1st class in the paper is apple, in our result its just around the baseline)
  - Is it due to our homemade way of generating trials?
- Given order is different, the distribution and trend of class-wise accuracy are very similar to the paper.
  - Around 1/2 classes performance are close to the baseline(25%), how does this happen?
- How to reproduce the exact generalization examples' score in the paper?
  - Though we are doing avg on 5 trials of each images among all seeds, we can't reach the same score as the paper.(e.g. 53.3%)
  - And our score is lower than the paper's score.
    

