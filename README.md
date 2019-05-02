<img src="https://github.com/ArturoDeza/NeuroFovea/blob/master/sample_gifs/Model_Diagram_New.png" width="900">

# Towards Metamerism via Foveated Style Transfer 
This repository containts the code to reproduce the Metamers used in the paper (Deza, Jonnalagadda, Eckstein. ICLR 2019). Link to the paper and discussion in openreview: https://openreview.net/forum?id=BJzbG20cFQ

This code has been tested successfully on CUDA version 8.0 (Ubuntu 14.04 and 16.04) and CUDA version 10.0 (Ubuntu 18.04).

The code to implement our model is mainly driven by:
* Adaptive Instance Normalization code: https://github.com/xunhuang1995/AdaIN-style
* pix2pix super-resolution module: https://github.com/phillipi/pix2pix
* The original Metamer code of Freeman & Simoncelli: https://github.com/freeman-lab/metamers
* A set of localized pooling regions stored in the *Receptive_Fields/* folder for the different rate of growth of the receptive fields specified by the scaling factor which should match the human psychophysical testing procedure as specified in the paper.

## What is a Metamer?
Metamers are a set of stimuli that are *physically different but perceptually indistinguishable to each other*. See below for an example.

| Input | Metamer | 
| --- | --- |
| <img src="https://github.com/ArturoDeza/NeuroFovea/blob/master/sample_gifs/1_color.png" width="440"> | <img src="https://github.com/ArturoDeza/NeuroFovea/blob/master/sample_gifs/Reference_vs_Synth_Metamer_V1_Color.gif" width="440"> | 

When maintaing center fixation on the orange dot the two images that are flipped back and forth should be perceptually indistinguishable to each other even though they are physically different (strong difference in the periphery *vs* the fovea).

### Rendering Metamers by varying receptive field size

<table>
  <tr>
    <td align="center"> Reference vs Synthesis Metamers (V1) </td>
    <td align="center"> Synthesis vs Synthesis Metamers (V2) </td>
  </tr>
    <td><img src="https://github.com/ArturoDeza/NeuroFovea/blob/master/sample_gifs/Reference_vs_Synth_Metamer_V1.gif" width="440"> </td>
    <td><img src="https://github.com/ArturoDeza/NeuroFovea/blob/master/sample_gifs/Synth_vs_Synth_Metamer_V2.gif" width="440"> </td>
  <tr>
    <td colspan="2"><b>Left:</b> we have a metamer that is metameric to its reference image. The rate of growth of the receptive fields on of the rendered metamer resembles the size of receptive fields of neurons in V1. <b>Right:</b>, we have two images that are heavily distorted in the visual periphery, are not metameric to the reference image, but are metameric to each other (perturbed with differente noise samples). The rate of growth of these receptive fields correspong to the sizes of V2 neurons, where it is hypothesized that the ventral stream is sensitive to texture. </td>
  </tr>
</table>

As in our previous demo, the metameric effects will only work properly if one fixates at the orange dot at the center of the image. In the paper we provide more details on how we psychophysically tested this phenomena using an eye-tracker to control for center fixation, viewing distance, display time, and the visual angle of the stimuli. We tested our model on grayscale images, and have extended the model in this code release to color images.

### Installation and pre-requisites for code functionality:
It was developed in CUDA 8.0 on Ubuntu 16.04 and has been tested on both CUDA 8.0 and CUDA 10.1 (though there might be some differences from CUDA 10.1 to 8.0) on Ubuntu 18.04. You will need to install:

[CUDA 10.1](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)

[CUDNN 7.5.1](https://developer.nvidia.com/rdp/cudnn-download)

[Torch (Lua)]( https://github.com/nagadomi/waifu2x/issues/253#issuecomment-445448928)

*Note: We are currently working on a pytorch re-implementation of our Metamer model. If you have one please let us know so we can post a link to your repo here as well.*

**The Full Dataset is also available here for future work in both grayscale and color Metamers,
they can be found in the *Datasets/* folder**

To complete the installation please run:

```
$ bash download_models_and_stimuli.sh
```

### Example code:

Generate a V1 metamer for the `512x512` image `10.png` with a center fixation, specified by the rate of growth of the receptive field: `s=0.25`

```
$ th NeuroFovea.lua -image Datasets/1_color.png -scale 0.25 -refinement 1 -color 1
```

To create a V2 metamer, change the scale from *0.25* to *0.5*. Scale is computed via receptive field size over retinal eccentricity of that receptive field and the values are only relevant given the size of the stimuli (26 x 26 degrees of visual angle rendered at 512 x 512 pixels). To compute the reference image, set the reference flag to 1.

Please read our paper to learn more about visual metamerism: https://openreview.net/forum?id=BJzbG20cFQ

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
