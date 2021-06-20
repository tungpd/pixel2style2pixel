import os

SAVE_PATH = '/home/tung/data/pretrained_models'
def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
    current_directory = os.getcwd()
    save_path = SAVE_PATH
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url


cmd = get_download_model_command('1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS', 'in-the-wild-images.zip')
cmd = get_download_model_command('1M24jfI-Ylb-k2EGhELSnxssWi9wGUokg', 'ffhq-r08.tfrecords')
print(cmd)
