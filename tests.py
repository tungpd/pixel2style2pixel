import face_detection
import editor

from PIL import Image
import numpy as np
import torch


checkpoint_path = "/home/tung/data/pretrained_models/psp_ffhq_encode.pt"


def test_load_from_checkpoint():
    """Check and summarise input and output dimensions"""
    latents = editor.load_latent_avg(checkpoint_path)
    assert tuple(latents.shape) == (18, 512)
    encoder = editor.load_encoder(checkpoint_path).to("cuda")
    input = torch.zeros((1, 3, 256, 256), device="cuda")
    output = encoder(input)
    assert tuple(output.shape) == (1, 18, 512)

    decoder = editor.load_decoder(checkpoint_path).to("cuda")
    input = torch.zeros((1, 18, 512), device="cuda")
    output, _ = decoder([input])
    assert tuple(output.shape) == (1, 3, 1024, 1024)
    print('test_load_from_checkpoint: SUCESSS')


test_load_from_checkpoint()

def test_face_detection_one_face():
    image_in = Image.open("test_data/face-ok.jpg")

    aligned_image, n_faces, quad = face_detection.align(image_in, face_index=0, output_size=256)

    assert aligned_image.size == (256, 256)
    assert n_faces == 1
    assert quad.shape == (4, 2)
    print("test_face_detection_one_face: SUCCESS")

test_face_detection_one_face()

def test_face_detection_multi_face():
    image_in = Image.open("test_data/two-face.jpg")

    aligned_image_1, n_faces_1, quad_1 = face_detection.align(image_in, face_index=0, output_size=256)
    aligned_image_2, n_faces_2, quad_2 = face_detection.align(image_in, face_index=1, output_size=256)

    assert aligned_image_1.size == aligned_image_2.size == (256, 256)
    assert n_faces_1 == n_faces_2 == 2
    assert (quad_1 != quad_2).all()
    print("test_face_detection_multi_face: SUCCESS")

test_face_detection_multi_face()

def test_composite():
    """Get the orginal back when compositing the same face"""
    image_in = Image.open("test_data/face-ok.jpg")
    output = image_in.copy()

    aligned_image, n_faces, quad = face_detection.align(image_in, face_index=0, output_size=1024)
    composited = face_detection.composite_images(quad, aligned_image, output)

    composited.thumbnail((128, 128))
    image_in.thumbnail((128, 128))
    assert np.allclose(np.array(composited), np.array(image_in), atol=100)
    print("test_composite: SUCCESS")

test_composite()

def test_composite_different_face():
    """Don't get the same image back"""
    image_in = Image.open("test_data/two-face.jpg")
    output = image_in.copy()

    aligned_image_1, n_faces_1, quad_1 = face_detection.align(image_in, face_index=0, output_size=1024)
    aligned_image_2, n_faces_2, quad_2 = face_detection.align(image_in, face_index=1, output_size=1024)

    composited = face_detection.composite_images(quad_1, aligned_image_2, output)

    composited.thumbnail((128, 128))
    image_in.thumbnail((128, 128))
    assert not np.allclose(np.array(composited), np.array(image_in), atol=100)
    print("test_composite_different_face: SUCCESS")
test_composite_different_face()
