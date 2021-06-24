# Copyright (c) 2021 Justin Pinkney

import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

from models.encoders import psp_encoders
from models.psp import get_keys, pSp
from models.stylegan2.model import Generator
from utils.common import log_input_image, tensor2im
import face_detection


tforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

device = "cuda"

idx_dict = {
    0: "conv_1",
    2: "conv_2",
    3: "conv_3",
    5: "conv_4",
    6: "conv_5",
    8: "conv_6",
    9: "conv_7",
    11: "conv_8",
    12: "conv_9",
    14: "conv_10",
    15: "conv_11",
}

# Edits from style space paper (S-index, channel-index, sense)
edits = {
    "smile": (6, 501, -1),
    "lipstick": (15, 45, -1),
    "goatee": (9, 421, -1),
    "sideburns": (12, 237, -1),
    "eye_size": (12, 110, 1),
    "gaze": (9, 409, 1),
    "mascara": (12, 414, -1),
    "squint": (14, 239, 1),
    "frown": (8, 28, -1),
    "eyebrow raise": (9, 340, 1),
    "eyebrow thickness": (12, 325, -1),
    "ear ring": (8, 81, 1),
    "grey hair": (11, 286, -1),
    "black hair": (12, 479, 1),
    "wavy hair": (6, 500, 1),
    "receding hair": (6, 364, 1),
    "masculinity": (9, 6, -1),
}


def load_latent_avg(checkpoint_path, device='cpu'):
    """Load the latent average used by encoder."""

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint["latent_avg"]


def load_encoder(checkpoint_path, device='cpu'):
    """Load the encoder portion from a model path"""

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    opts = checkpoint['opts']
    opts['checkpoint_path'] = checkpoint_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts['n_styles'] = int(math.log(opts['output_size'], 2)) * 2 - 2

    opts['device'] = device

    opts = Namespace(**opts)
    if opts.encoder_type == 'GradualStyleEncoder':
        encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', opts)
    elif opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
        encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', opts)
    elif opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
        encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', opts)
    else:
        raise Exception('{} is not a valid encoder'.format(opts.encoder_type))

    encoder.load_state_dict(get_keys(checkpoint, "encoder"), strict=True)

    return encoder

def load_decoder(checkpoint_path):
    """Load the decoder portion from a model path"""

    checkpoint = torch.load(checkpoint_path)
    decoder = Generator(1024, 512, 8)
    decoder.load_state_dict(get_keys(checkpoint, "decoder"), strict=True)

    return decoder

def load_model(checkpoint_path):
    """Load all the parts of the model"""
    encoder = load_encoder(checkpoint_path).to(device)
    decoder = load_decoder(checkpoint_path).to(device)
    latent_avg = load_latent_avg(checkpoint_path).to(device)

    return encoder, decoder, latent_avg


def run(encoder, decoder, latent_avg, original):
    """Encode and decode an image"""

    input_image = tforms(original).to(device)

    with torch.no_grad():

        codes = encoder(input_image.unsqueeze(0).float())
        codes = codes + latent_avg.repeat(codes.shape[0], 1, 1)
        image, latent = decoder([codes], input_is_latent=True)
        out_im = image.squeeze()

    return tensor2im(out_im)


class StyleManipulator:
    """Allows manipulation of Style latent space"""

    def __init__(self, device="cuda"):
        # Edits is a dictionary structured:
        # {conv_layer_name: {idx1: val1, idx2: val2}}
        self.edits = {}
        self.device = device

    def get_hook(self, name):
        """Returns hook to modify output of conv layers"""
        def hook(module, input, output):
            if name in self.edits:
                this_edit = self.edits[name]
                for channel in this_edit.keys():
                    output[0, channel] += this_edit[channel]
                return output
        return hook


def manipulate_model(g):
    """Add hooks to the StyleGAN generator"""
    hooks = []
    manipulator = StyleManipulator()

    layer = g.conv1.conv.modulation
    hook = manipulator.get_hook("conv_1")
    hooks.append(layer.register_forward_hook(hook))

    for idx, styledConv in enumerate(g.convs, 2):
        layer = styledConv.conv.modulation
        hook = manipulator.get_hook(f"conv_{idx}")
        hooks.append(layer.register_forward_hook(hook))

    return manipulator

