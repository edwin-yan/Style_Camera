# Style Camera CycleGAN

Apply Neural Style Transfer on Camera Images.

Please be advised that this project used the CycleGAN model from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

## Environment
There are two ways to reproduce my environment:
1. **conda_env.yml** - Conda Environment Export
2. **requirements.txt** - Pip Packages Export
_**Disclaimer**: This code is developed and tested on Windows Environment, although it should work on Linux as well_

## Quick Start
`
python main.py --style cezanne --resolution 720p
`

_Although it may work on CPU technically, it requires a lot of processing power inference on live camera (even just 10-20 FPS). Therefore, **CUDA CuDNN** is required to run the program. If your GPU is not powerful enough, please use 720P or 480P resolution._
