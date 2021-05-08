# Style Camera CycleGAN

Apply Neural Style Transfer on live web camera streams. Styles from Cezanne, Monet, Ukiyoe and Vangogh are provided.

Please be advised that this project uses the CycleGAN model from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

## Environment
There are two ways to reproduce my environment:
1. **conda_env.yml** - Conda Environment Export
2. **requirements.txt** - Pip Packages Export

_**Disclaimer**: This code is developed and tested on Windows 10 Pro, although it should work on the Linux distributions as well_

## Quick Start
`
python main.py 
`

**Optional Parameters:**
- **style:** Options are cezanne, monet, ukiyoe and vangogh. Default is cazanne.
- **resolution:** options are 480p, 720p and 1080p. Default is 720p.
- **fps:** if FPS is set to be less than 30, then the program will limit the fps. Otherwise, it will use highest possible fps. Default is unlimited.
- **force_cpu:** not recommended! CPU will be used if it is set to be True. Default is False.

Since it requires a lot of processing power to apply neural style transfer on live camera even with low FPS,  **CUDA CuDNN** is recommended. If you want to run on CPU only, please set --force_cpu True. Please note that the FPS is less or equal to 1 frame per second running on AMD Threadripper 2970X 48Threads@4GHz.

If CUDA is used, 480P resolution requires less than 2.5G dedicated GPU Memory. 720P requires about 3.5G dedicated GPU Memory. 1080P Requires at least 5.5G dedicated GPU Memory. Therefore, please use 720P or 480P resolution, if your GPU is not powerful enough.
