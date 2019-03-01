# Towards Metamerism via Foveated Style Transfer 
Code to reproduce the Metamers used in the paper (Deza, Jonnalagadda, Eckstein. ICLR 2019). Link to the paper and discussion in openreview: https://openreview.net/forum?id=BJzbG20cFQ

Code has been tested successfully on CUDA version 8.0 (Ubuntu 14.04 and 16.04) and CUDA version 10.0 (Ubuntu 18.04).

The backbone of this code is mainly driven by:
* Adaptive Instance Normalization code: https://github.com/xunhuang1995/AdaIN-style
* pix2pix super-resolution module: https://github.com/phillipi/pix2pix
* a mixture of localized foveated templates stored in the main folder for the different rate of growth of the receptive fields specified by the scaling factor which should match the human psychophysical testing procedure as specified in the paper.

### Example code:

Generate a metamer for the `512x512` image `10.png` with a center fixation, specified by the rate of growth of the receptive field: `s=0.25`

```
$ th NeuroFoveaAlpha.lua -image 10.png -scale 0.25
```

Then run the pix2pix refinement module:

```
DATA_ROOT=./datasets/Metamers name=Metamers which_direction=AtoB phase=test th test.lua
```


__Observation:__ Id you'd like to use another scaling factor, as well as change the point of fixation, you have to change the `-scale` parameter and potentially create a new 'window' folder. To generate metamers that match the rate of growth of the receptive field size of V1, we need to set the scale factor to 0.25. To generate metamers that match the rate of growth of the R.F.'s in V2, we need to set the scale factor to 0.5. It's worth noting that in our experiments our psychophysical evaluations are done both against the compressed image (ran through the auto-encoder) which is a close approximation to the high-resolution original gray scale image, as well as against synthesized metamers (Synth vs Synth). Other implementations implementations of AdaIN as well as different style transfer models may improve the general NeuroFovea metamer generation pipeline of localized Auto-Style Transfer.

The model currently works for grayscale images.
