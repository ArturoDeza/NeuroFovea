<img src="https://github.com/ArturoDeza/NeuroFovea/blob/master/Model_Diagram_New.png" width="900">

# Towards Metamerism via Foveated Style Transfer 
This repository containts the code to reproduce the Metamers used in the paper (Deza, Jonnalagadda, Eckstein. ICLR 2019). Link to the paper and discussion in openreview: https://openreview.net/forum?id=BJzbG20cFQ

This code has been tested successfully on CUDA version 8.0 (Ubuntu 14.04 and 16.04) and CUDA version 10.0 (Ubuntu 18.04).

The code to implement our model is mainly driven by:
* Adaptive Instance Normalization code: https://github.com/xunhuang1995/AdaIN-style
* pix2pix super-resolution module: https://github.com/phillipi/pix2pix
* The original Metamer code of Freeman & Simoncelli: https://github.com/freeman-lab/metamers
* A mixture of localized foveated templates stored in the *Datasets/* for the different rate of growth of the receptive fields specified by the scaling factor which should match the human psychophysical testing procedure as specified in the paper.

## What is a Metamer?

Metamers are a set of stimuli that are *physically different but perceptually indistinguishable to each other*. For example, on the **left** we have a metamer that is metameric to its reference image (an approximate to the original high resolution image when sent through an autoencoder). On the **right**, we have two images that are heavily distorted in the visual periphery, are not metameric to the reference image, but are metameric to each other (perturbed with differente noise samples). For the metameric effects to work properly one must fixate at the orange dot at the center of the image. In the paper we provide more details on how we psychophysically tested this phenomena using an eye-tracker to control for center fixation, viewing distance and the visual angle of the display.

| Reference vs Synthesis Metamers (V1) | Synthesis vs Synthesis Metamers (V2) |
| --- | --- |
| <img src="https://github.com/ArturoDeza/NeuroFovea/blob/master/Reference_vs_Synth_Metamer_V1.gif" width="440"> | <img src="https://github.com/ArturoDeza/NeuroFovea/blob/master/Synth_vs_Synth_Metamer_V2.gif" width="440"> |

### Pre-requisites for code functionality:
It was developed in CUDA 8.0 on Ubuntu 16.04 and has been tested on both CUDA 8.0 and CUDA 10.1 (though there might be some differences from CUDA 10.1 to 8.0) on Ubuntu 18.04. You will need to install:

[CUDA 10.1](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)

[CUDNN 7.5.1](https://developer.nvidia.com/rdp/cudnn-download)

[Torch (Lua)]( https://github.com/nagadomi/waifu2x/issues/253#issuecomment-445448928)

*Note: We are currently working on a pytorch re-implementation of our Metamer model. If you have one please let us know so we can post a link to your repo here as well.*


### Example code:

#### Create a V1 Metamer

Generate a metamer for the `512x512` image `10.png` with a center fixation, specified by the rate of growth of the receptive field: `s=0.25`

```
$ th NeuroFoveaAlpha.lua -image Datasets/10.png -scale 0.25
```

#### Create a V2 Metamer

```
$ th NeuroFoveaAlpha.lua -image Datasets/5.png -scale 0.5
```

#### Create a V2 Metamer with pix2pix refinement (as produced in the paper):

```
$ th NeuroFoveaAlpha.lua -image Datasets/8.png -scale 0.5 - superresolution 1
```

#### Create a Color V1 Metamer

```
$ th NeuroFoveaAlpha_Color.lua -image Datasets/10.png -scale 0.25
```

#### Metamerize a folder of images (Fast Metamer Generation process for big datasets): 

### The Full Dataset is also available here for future work in both grayscale and color Metamers,
they can be found in the *Datasets/* folder

__Observation:__ Id you'd like to use another scaling factor, as well as change the point of fixation, you have to change the `-scale` parameter and potentially create a new 'window' folder. To generate metamers that match the rate of growth of the receptive field size of V1, we need to set the scale factor to 0.25. To generate metamers that match the rate of growth of the R.F.'s in V2, we need to set the scale factor to 0.5. It's worth noting that in our experiments our psychophysical evaluations are done both against the compressed image (ran through the auto-encoder) which is a close approximation to the high-resolution original gray scale image, as well as against synthesized metamers (Synth vs Synth). Other implementations implementations of AdaIN as well as different style transfer models may improve the general NeuroFovea metamer generation pipeline of localized Auto-Style Transfer. 

The model has been psychophysically tested on grayscale images, although it works approximately well on color images. 

We hope this code and our paper can help researchers, scientists and engineers improve the use and design of metamer models that have potentially exciting applications in both computer vision and visual neuroscience.

This code is free to use for Research Purposes, and if used/modified in any way please consider citing:

```
@inproceedings{
deza2018towards,
title={Towards Metamerism via Foveated Style Transfer},
author={Arturo Deza and Aditya Jonnalagadda and Miguel P. Eckstein},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=BJzbG20cFQ},
}
```

Other inquiries: arturo_deza@fas.harvard.edu
