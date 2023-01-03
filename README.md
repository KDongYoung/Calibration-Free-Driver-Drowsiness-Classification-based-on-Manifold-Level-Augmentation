# Calibration-Free Driver Drowsiness Classification based on Manifold-Level Augmentation 
This is an official repo for Calibration-Free Driver Drowsiness Classification based on Manifold-Level Augmentation (Proc. Int. Winter Conf. Brain-Computer Interface, 2023) [\[Paper\]](https://arxiv.org/abs/2212.13887)

## Description

We propose a robust calibration-free driver drowsiness classification framework by a manifold-level augmentation. We gathered samples from  all the subjects/domains in a domain-balaced and class-balanced manner an composed a mini-batch. We generated samples of unseen domains by mixing intermediate instances’ style statistics to preserve the characteristics of EEG signals.

## Getting Started

### Environment Requirement

Clone the repo:

```bash
git clone https://github.com/KDongYoung/Calibration-Free-Driver-Drowsiness-Classification-based-on-Manifold-Level-Augmentation.git
```

Install the requirements using `conda`:

```terminal
conda create -n EEG_AUG python=3.8.13
conda activate EEG_AUG
pip install -r requirements.txt
```

IF using a Docker, use the recent image file ("pytorch:22.04-py3") uploaded in the [\[NVIDIA pytorch\]](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) when running a container


## Data preparation

First, create a folder `${DATASET_DIR}` to store the data of each subject.

Download the unbalanced dataset available in the paper titled "EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable Convolutional Neural Network" published in *IEEE Transactions on Neural Networks and Learning Systems* at 2022.

(Ref: J. Cui, Z. Lan, O. Sourina and W. Müller-Wittig, "EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable Convolutional Neural Network," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2022.3147208.)

Unbalanced dataset available in [\[Dataset Download Link]\](https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset_unbalanced_/16586957)

The directory structure should look like this:

```
${DATASET_DIR}
	|--${unbalanced_dataset.mat}
```

### Training from scratch

```shell script
# train
python TotalMain.py --mode train
# test
python TotalMain.py --mode infer
```

The (BEST model for each SUBJECT and the tensorboard records) are saved in `${MODEL_SAVE_DIR}/{seed}_{step}/{model_name}` by default

The results are saved in text and csv files in `${MODEL_SAVE_DIR}/{seed}_{step}/{Results}/{evalauation metric}` by default

-> The BEST models are saved separately in each folder based on the evaluation metric used to select the model for validation.

The result directory structure would look like this:

```
${MODEL_SAVE_DIR}
    ${seed}_{step}
	|--${model_name}
	    |--${models}
	    	|--${evaluation metric}
	    |--${tensorboard records}
        |--${Results}
	    |--${evaluation metric}
	    	|--${csv file}
		|--${txt file}
```

### Evaluation

**The average results (%) for drowsiness classification:**
| Model                      | F1-score ± Std. | 
| -------------------------- | --------------- | 
| Baseline                   |  64.67 ± 18.26  | 
| Baseline + augmentation (ours)   |  68.41 ± 15.28  | 

Std.: Standard deviation

## Citation

```
@article{kim2022calibration,
  title={Calibration-Free Driver Drowsiness Classification based on Manifold-Level Augmentation},
  author={Kim, Dong-Young and Han, Dong-Kyun and Shin, Hye-Bin},
  journal={arXiv preprint arXiv:2212.13887},
  year={2022}
}
```
