# Towards Metamerism via Foveated Style Transfer 
Code to reproduce the Metamers used in the paper (Deza, Jonnalagadda, Eckstein. ICLR 2019). Link to the paper and discussion in openreview: https://openreview.net/forum?id=BJzbG20cFQ

Code has been tested successfully on Ubuntu 14.04, 16.04, and 18.04 with CUDA version 8.0 (Ubuntu 14.04 and 16.04) and 10.0 (Ubuntu 18.04)

The backbone of this code is mainly dirven by:
* Adaptive Instance Normalization code: https://github.com/xunhuang1995/AdaIN-style
* pix2pix super-resolution module: https://github.com/phillipi/pix2pix
* a mixture of localized foveated templates stored in the main folder for the different rate of growth of the receptive fields specified by the scaling factor which should match the human psychophysical testing procedure as specified in the paper.

README.md file written by Arturo Deza.

### Generate Metamers for any of the a sample images:

```
$ th NeuroFoveaAlpha.lua -image 10.png -scale 0.25
```

Observation: Id you'd like to use another scaling factor, as well as change the point of fixation, you have to change the `-scale` parameter and potentially create a new 'window' folder.
