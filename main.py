import model
import torch
from functools import partial
import torch.nn as nn
from util import get_transform_simplified
import cv2
from PIL import Image
import time
import argparse


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def inv_normalize(img):
    # Adding 0.1 to all normalization values since the model is trained (erroneously) without correct de-normalization
    mean = torch.Tensor([0.5, 0.5, 0.5]).to(device)
    std = torch.Tensor([0.5, 0.5, 0.5]).to(device)

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    return img


def transfer_style(image):
    transformed_image = get_transform_simplified()(image)
    image_device = transformed_image.unsqueeze(0).to(device)

    with torch.no_grad():
        generated_images = netG(image_device)

    generated_image = inv_normalize(generated_images).cpu()[0]
    return generated_image.numpy().transpose(1, 2, 0)


def load_model(pretrained_dir, device):
    print("Loading Model....")
    norm = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    netG = model.ResnetGenerator(input_nc=3,
                                 output_nc=3,
                                 ngf=64,
                                 norm_layer=norm,
                                 use_dropout=False,
                                 n_blocks=9
                                 ).to(device)
    netG.eval()

    state_dict = torch.load(pretrained_dir, map_location=device)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, netG, key.split('.'))
    netG.load_state_dict(state_dict)
    print("Model Loaded...")
    return netG

def determine_resolution(arg):
    def make_1080p(cap):
        cap.set(3, 1920)
        cap.set(4, 1080)

    def make_720p(cap):
        cap.set(3, 1280)
        cap.set(4, 720)

    def make_480p(cap):
        cap.set(3, 640)
        cap.set(4, 480)

    if arg == '1080p':
        return make_1080p
    elif arg == '720p':
        return make_720p
    else:
        return make_480p


def style_camera():
    print('Initializing Camera...this may take a few seconds...')
    vc = cv2.VideoCapture(0)
    set_resolution(vc)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    frame_rate, prev = 10, 0

    while rval:
        cv2.imshow("preview", frame)
        time_elapsed = time.time() - prev
        rval, frame = vc.read()

        key = cv2.waitKey(40)
        if key == 27:  # exit on ESC
            break

        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image)
            new_image = transfer_style(img_pil)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('style', new_image)

    cv2.destroyWindow("preview")
    cv2.destroyWindow("style")


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='cezanne', help='Styles to apply. Options are cezanne, monet, ukiyoe and vangogh')
    parser.add_argument('--resolution', type=str, default='720p', help='Camera Resolution. Options are 480P, 720P and 1080P. WARNING: 1080P may require high-end GPU')
    args = parser.parse_args()
    print(f'Config:\nStyle: {args.style}\nCamera Resolution: {args.resolution}')
    model_path = f'./styles/style_{args.style}.pth'
    return model_path, determine_resolution(args.resolution)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA Device Detected. Use CUDA...")
        pretrained_dir, set_resolution = process_args()
        device = torch.device('cuda')
        netG = load_model(pretrained_dir, device)
        style_camera()
    else:
        print("Although it is possible to run on CPU technically, it may be very slow and laggy. Please run this program on CUDA device")
        exit(1)
