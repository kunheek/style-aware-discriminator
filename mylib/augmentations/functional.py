import math

import torch
import torchvision.transforms.functional as VF


def pad_random_affine(img, angle, translate, scale, shear, interpolation):
    """
        img (PIL Image or Tensor): Image to be transformed.

    Returns:
        PIL Image or Tensor: Affine transformed image.
    """

    img_size = VF.get_image_size(img)
    
    if (torch.rand(1) < 0.8) and angle != 0.0:
        angle = torch.empty(1).uniform_(-angle, angle)
        angle = float(angle.item())
    else:
        angle = 0.0
    
    if (torch.rand(1) < 0.8) and (translate != (0, 0)):
        max_dx = float(translate[0] * img_size[0])
        max_dy = float(translate[1] * img_size[1])
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        translations = (tx, ty)
    else:
        translations = (0, 0)

    if (torch.rand(1) < 0.8) and not isinstance(scale, float):
        scale = float(torch.empty(1).uniform_(*scale).item())
    else:
        scale = 1.0

    shear_x = shear_y = 0.0
    if (torch.rand(1) < 0.8) and (shear != (0, 0)):
        x = shear[0]
        shear_x = float(torch.empty(1).uniform_(-x, x).item())
        if shear[1] != 0:
            shear_y = float(torch.empty(1).uniform_(-shear[1], shear[1]).item())
        shear = (shear_x, shear_y)

    pad = round(math.sin(math.radians(abs(angle))) * max(img_size))
    pad += max(abs(translations[0]), abs(translations[1]))
    if scale < 1.0:
        pad += round(max(img_size) * (1.0 - scale) * 0.5)
    if shear != (0.0, 0.0):
        pad += round(math.tan(math.radians(max(shear))) * max(img_size))

    # pad = left, top, right, bottom
    img = VF.pad(img, pad, padding_mode="reflect")
    img = VF.affine(img, angle, translations, scale, shear, interpolation)
    # crop = top, left, height, width
    img = VF.crop(img, pad, pad, img_size[1], img_size[0])
    return img
