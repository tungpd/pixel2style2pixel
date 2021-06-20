from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp
from gdrive_download import get_download_model_command


experiment_type = 'ffhq_encode'
CODE_DIR = '/home/tung/repos/pixel2style2pixel'
PRETRAINED_MODEL_DIR = '/home/tung/data/pretrained_models'

MODEL_PATHS = {
    "ffhq_encode": {"id": "1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0", "name": "psp_ffhq_encode.pt"},
    "ffhq_frontalize": {"id": "1_S4THAzXb-97DbpXmanjHtXRyKxqjARv", "name": "psp_ffhq_frontalization.pt"},
    "celebs_sketch_to_face": {"id": "1lB7wk7MwtdxL-LL4Z_T76DuCfk00aSXA", "name": "psp_celebs_sketch_to_face.pt"},
    "celebs_seg_to_face": {"id": "1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz", "name": "psp_celebs_seg_to_face.pt"},
    "celebs_super_resolution": {"id": "1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu", "name": "psp_celebs_super_resolution.pt"},
    "toonify": {"id": "1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz", "name": "psp_ffhq_toonify.pt"}
}


# path = MODEL_PATHS[experiment_type]
# download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
# print(download_command)

# stream = os.popen(download_command)
# print(stream.read())

EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        # "model_path": "pretrained_models/psp_ffhq_encode.pt",
        "model_path": "/home/tung/data/pretrained_models/psp_ffhq_toonify.pt",
        # "image_path": "notebooks/images/input_img.jpg",
        "image_path": "notebooks/images/IMG_20210320_110540.jpg",
        # "image_path": "/home/tung/data/datasets/CelebAMask-HQ/test_img/18.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "ffhq_frontalize": {
        "model_path": "pretrained_models/psp_ffhq_frontalization.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celebs_sketch_to_face": {
        "model_path": "pretrained_models/psp_celebs_sketch_to_face.pt",
        "image_path": "notebooks/images/input_sketch.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    },
    "celebs_seg_to_face": {
        "model_path": "pretrained_models/psp_celebs_seg_to_face.pt",
        "image_path": "notebooks/images/input_mask.png",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.ToOneHot(n_classes=19),
            transforms.ToTensor()])
    },
    "celebs_super_resolution": {
        "model_path": "pretrained_models/psp_celebs_super_resolution.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[16]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "toonify": {
        "model_path": "pretrained_models/psp_ffhq_toonify.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
    raise ValueError("Pretrained model was unable to be downlaoded correctly!")

model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']
opts['device'] = 'cpu'
print(opts)

opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False
if 'output_size' not in opts:
    opts['output_size'] = 1024

opts = Namespace(**opts)
net = pSp(opts)
net.eval()
# net.cuda()
print('Model successfully loaded!')

image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
original_image = Image.open(image_path)
if opts.label_nc == 0:
    original_image = original_image.convert("RGB")
else:
    original_image = original_image.convert("L")


# original_image.resize((256, 256))

def download_face_landmarks():
  if not os.path.exists('{}/shape_predictor_68_face_landmarks.dat.bz2'.format(PRETRAINED_MODEL_DIR)):
    stream = os.popen('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P {}'.format(PRETRAINED_MODEL_DIR))
    print(stream.read())
  stream = os.popen('bzip2 -dk {}/shape_predictor_68_face_landmarks.dat.bz2'.format(PRETRAINED_MODEL_DIR))
  print(stream.read())

def run_alignment(image_path):
  import dlib
  from scripts.align_all_parallel import align_face
  if not os.path.exists('{}/shape_predictor_68_face_landmarks.dat'.format(PRETRAINED_MODEL_DIR)):
      download_face_landmarks()
  predictor = dlib.shape_predictor("{}/shape_predictor_68_face_landmarks.dat".format(PRETRAINED_MODEL_DIR))
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image

if experiment_type not in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
  input_image = run_alignment(image_path)
else:
  input_image = original_image

# input_image.resize((512, 512))

img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image)

def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to("cpu").float(), randomize_noise=False, resize=False)
        # result_batch = net(inputs.to("cpu").float(), randomize_noise=False)
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cpu"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cpu").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch

if experiment_type in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
    latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
else:
    latent_mask = None

with torch.no_grad():
    tic = time.time()
    result_image = run_on_batch(transformed_image.unsqueeze(0), net, latent_mask)[0]
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))
input_vis_image = log_input_image(transformed_image, opts)
print('result_image size: %s, %s, %s'%result_image.shape)
output_image = tensor2im(result_image)
output_image.save(open('output_image.jpg', 'w'), 'JPEG')
if experiment_type == "celebs_super_resolution":
    res = np.concatenate([np.array(input_image.resize((256, 256))),
                          np.array(input_vis_image.resize((256, 256))),
                          np.array(output_image.resize((256, 256)))], axis=1)
else:
    res = np.concatenate([np.array(input_vis_image.resize((256, 256))),
                          np.array(output_image.resize((256, 256)))], axis=1)

res_image = Image.fromarray(res)

res_image.show()

