# Unsupervised Audio Source Separation using Generative Priors

Refer to diagram below for a summary of the approach.

![Proposed Approach for Source Separation](https://github.com/vivsivaraman/Metric-Learning-Pipelines-for-Speaker-Diarization/blob/master/approach.png)

## Requirements
* python 3.6
* numpy >= 1.11.0
* tensorflow = 1.12.0
* scikit-learn >= 0.18
* matplotlib >= 2.1.0
* librosa

## Prerequisites
The Digit, Drum and Piano audio datasets can be downloaded from: https://github.com/chrisdonahue/wavegan

The datasets need to be in the directory ``` /data ```

## Data Processing

Run ```create_dataset.py``` with appropriate arguments to randomly sample 1000 examples of digit, drums and piano audio from the respective datasets and form the normalized mixtures. 


### Pre-trained GAN Priors
Download the Pre-trained WaveGAN Priors from and save the contents in the respective folders in the ``` /ckpts ``` directory 

### Perform Source Separation
Run ```run_twosourcesep.py``` to perform two source separation on the appropriate mixture combination (digit-drums, digit-piano or drums-piano). 

To specify the paths for the pre-trained models, mixture data, results, type of sources and other hyperparameters,modify the arguments of ```run_twosourcesep.py``` 

Run ```run_threesourcesep.py``` to perform three source separation on the digit-drums-piano mixture combination 

To specify the paths for the pre-trained models, mixture data, results, type of sources and other hyperparameters,modify the arguments of  ```run_threesourcesep.py``` 


## Citations

Our paper is cited as:

```

```
