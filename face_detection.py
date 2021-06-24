import dlib
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from scipy.ndimage import gaussian_filter
import cv2


MODEL_PATH = "/home/tung/data/pretrained_models/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()

def align(image_in, face_index=0, output_size=256):
    landmarks = list(get_landmarks(image_in))
    n_faces = len(landmarks)
    face_index = min(n_faces-1, face_index)
    if n_faces == 0:
        aligned_image = image_in
        quad = None
    else:
        aligned_image, quad = image_align(image_in, landmarks[face_index], output_size=output_size)

    return aligned_image, n_faces, quad


def composite_images(quad, img, output):
    """Composite an image into and output canvas according to transformed co-ords"""
    output = output.convert("RGBA")
    img = img.convert("RGBA")
    input_size = img.size
    src = np.array(((0, 0), (0, input_size[1]), input_size, (input_size[0], 0)), dtype=np.float32)
    dst = np.float32(quad)
    mtx = cv2.getPerspectiveTransform(dst, src)
    img = img.transform(output.size, Image.PERSPECTIVE, mtx.flatten(), Image.BILINEAR)
    output.alpha_composite(img)

    return output.convert("RGB")


def get_landmarks(image):
    """Get landmarks from PIL image"""
    shape_predictor = dlib.shape_predictor(MODEL_PATH)

    max_size = max(image.size)
    reduction_scale = int(max_size/512)
    if reduction_scale == 0:
        reduction_scale = 1
    downscaled = image.reduce(reduction_scale)
    img = np.array(downscaled)
    detections = detector(img, 0)

    for detection in detections:
        try:
            face_landmarks = [(reduction_scale*item.x, reduction_scale*item.y) for item in shape_predictor(img, detection).parts()]
            yield face_landmarks
        except Exception as e:
            print(e)

def image_align(src_img, face_landmarks, output_size=512, transform_size=2048, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
    # Align function modified from ffhq-dataset
    # See https://github.com/NVlabs/ffhq-dataset for license

    lm = np.array(face_landmarks)
    lm_eye_left      = lm[2:3]  # left-clockwise
    lm_eye_right     = lm[0:1]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = 0.71*(eye_right - eye_left)
    mouth_avg    = lm[4]
    eye_to_mouth = 1.35*(mouth_avg - eye_avg)

    # Choose oriented crop rectangle.
    x = eye_to_eye.copy()
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    quad_orig = quad.copy()
    qsize = np.hypot(*x) * 2

    img = src_img.convert('RGBA').convert('RGB')

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        if alpha:
            mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
            mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
            img = np.concatenate((img, mask), axis=2)
            img = Image.fromarray(img, 'RGBA')
        else:
            img = Image.fromarray(img, 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    return img, quad_orig
