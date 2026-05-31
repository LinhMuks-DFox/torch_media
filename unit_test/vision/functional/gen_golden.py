#!/usr/bin/env python3
"""Golden reference values for the vision functional unit tests, from torchvision 0.20.1.
Run in the project venv:  .venv/bin/python unit_test/vision/functional/gen_golden.py
Image: [3,4,4] float in [0,1]."""
import torch
import torchvision.transforms.functional as F


def show(name, value):
    print(f"{name} = {value}")


img = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 47.0

show("hflip[0,0,0] / [0,0,3]", [round(float(F.hflip(img)[0, 0, 0]), 4), round(float(F.hflip(img)[0, 0, 3]), 6)])
show("vflip sum", round(float(F.vflip(img).sum()), 4))
show("rgb_to_grayscale[0,0,0]", round(float(F.rgb_to_grayscale(img)[0, 0, 0]), 6))
show("normalize(0.5,0.5)[0,0,0]", round(float(F.normalize(img, [0.5] * 3, [0.5] * 3)[0, 0, 0]), 6))
show("center_crop([2,2]) sum", round(float(F.center_crop(img, [2, 2]).sum()), 4))
show("adjust_brightness(1.5)[0,0,3]", round(float(F.adjust_brightness(img, 1.5)[0, 0, 3]), 6))
show("invert[0,0,0]", round(float(F.invert(img)[0, 0, 0]), 6))
