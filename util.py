import torchvision.transforms as transforms
from PIL import Image

def __make_power_2(img, base, method=Image.BICUBIC):
    """
    Resize the image

    Parameters
    ----------
    img    - the input image
    base   - the base to be powered on
    method - the method to rescale the image

    Returns
    -------
    if the height and width are already divisible by the base, return original image, otherwise, resize the image

    """
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __print_size_warning(ow, oh, w, h):
    """
    Print the warning image. It should only print once becasue the image will be resize multiple times every second (depends on the FPS).

    Parameters
    ----------
    ow  - Original Width
    oh  - Original Height
    w   - Width
    h   - Height

    Returns
    -------
    """

    if not hasattr(__print_size_warning, 'has_printed'):
        print(f"The image size needs to be a multiple of 4. The loaded image size was ({ow}, {oh}), so it was adjusted to ({w}, {h}). This adjustment will be done to all images whose sizes are not multiples of 4")
        __print_size_warning.has_printed = True

def get_transform(method=Image.BICUBIC, convert=True):
    """
    Prepare Images to feed into the model. This is step is necessary to match to the training data used for the model.
    It is simplified based on: https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/f13aab8148bd5f15b9eb47b690496df8dadbab0c/data/base_dataset.py

    Parameters
    ----------
    method      - the resize method
    convert     - whether to normalize image

    Returns
    -------
    Return a list of jobs to apply
    """

    transform_list = [transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))]
    if convert:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

