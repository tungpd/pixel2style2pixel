from pixel2style2pixel import editor
from pixel2style2pixel import face_detection
from PIL import Image


edit_controls = {k: 0 for k in editor.edits.keys()}

inputs = {'original': '/home/tung/repos/pixel2style2pixel/notebooks/images/IMG_20210320_110540.jpg'}

inputs.update(edit_controls)

outputs = {'image': ''}

opts = {'checkpoint': '/home/tung/data/pretrained_models/psp_ffhq_encode.pt',
        'face_detector': '/home/tung/data/pretrained_models/shape_predictor_5_face_landmarks.dat'}

def init(opts):
    checkpoint_path = opts['checkpoint']
    face_detection.MODEL_PATH = opts['face_detector']

    encoder, decoder, latent_avg = editor.load_model(checkpoint_path)

    manipulator = editor.manipulate_model(decoder)
    manipulator.edits = {editor.idx_dict[v[0]]: {v[1]: 0} for k, v in editor.edits.items()}

    return encoder, decoder, latent_avg, manipulator

def generate(encoder, decoder, latent_avg, manipulator, input_args):
    original = Image.open(input_args['original'])
    input_size = [1024, 1024]

    output = original.copy()
    cropped, n_faces, quad = face_detection.align(original)

    for k, v in editor.edits.items():
        layer_index, channel_index, sense = v
        conv_name = editor.idx_dict[layer_index]
        manipulator.edits[conv_name][channel_index] = input_args[k]*sense
        for i in range(n_faces):
            if i > 0:
                cropped, _, quad = face_detection.align(original, face_index=i)
            transformed_crop = editor.run(encoder, decoder, latent_avg, cropped)
            output = face_detection.composite_images(quad, transformed_crop, output)
    return output

if __name__ == '__main__':
    encoder, decoder, latent_avg, manipulator = init(opts)
    output = generate(encoder, decoder, latent_avg, manipulator, inputs)
    output.show()
