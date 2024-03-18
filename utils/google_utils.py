# Google utils: https://cloud.google.com/storage/docs/reference/libraries

import os
import platform
import subprocess
import time
from pathlib import Path

import requests
import torch


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes

def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    file = Path(weights).name

    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']  # available models

    if file in models and not os.path.isfile(weights):
        # Google Drive
        # d = {'yolov5s.pt': '1R5T6rIyy3lLwgFXNms8whc-387H0tMQO',
        #      'yolov5m.pt': '1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr',
        #      'yolov5l.pt': '1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV',
        #      'yolov5x.pt': '1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS'}
        # r = gdrive_download(id=d[file], name=weights) if file in d else 1
        # if r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6:  # check
        #    return

        try:  # GitHub
            url = 'https://github.com/ultralytics/yolov5/releases/download/v2.0/' + file
            print('Downloading %s to %s...' % (url, weights))
            if platform.system() == 'Darwin':  # avoid MacOS python requests certificate error
                r = os.system('curl -L %s -o %s' % (url, weights))
            else:
                torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            print('Download error: %s' % e)
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            print('Downloading %s to %s...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))  # torch.hub.download_url_to_file(url, weights)
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # check
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                print('ERROR: Download failure: %s' % msg)
            print('')
            return

def gdrive_download(id='', file='tmp.zip'):
    # Downloads a file from Google Drive. from yolov7.utils.google_utils import *; gdrive_download()
    t = time.time()
    file = Path(file)
    cookie = Path('cookie')  # gdrive cookie
    print(f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ', end='')
    file.unlink(missing_ok=True)  # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists('cookie'):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == '.zip':
        print('unzipping... ', end='')
        os.system(f'unzip -q {file}')  # unzip
        file.unlink()  # remove zip to free space

    print(f'Done ({time.time() - t:.1f}s)')
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""

# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
