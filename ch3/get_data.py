import os
import shutil
import urllib.request
import zipfile

BASE_URL = "https://storage.googleapis.com/learning-datasets/"
TRAINING_ZIP = "horse-or-human.zip"
VALIDATION_ZIP = "validation-horse-or-human.zip"


def download_training_data():
    url = BASE_URL + TRAINING_ZIP
    dir = 'horse-or-human/training/'

    if os.path.exists(dir):
        shutil.rmtree(dir)

    print("Downloading data from", url)
    urllib.request.urlretrieve(url, TRAINING_ZIP)

    with zipfile.ZipFile(TRAINING_ZIP, 'r') as zip_ref:
        zip_ref.extractall(dir)

    os.remove(TRAINING_ZIP)


def download_validation_data():
    url = BASE_URL + VALIDATION_ZIP
    dir = 'horse-or-human/validation/'

    if os.path.exists(dir):
        shutil.rmtree(dir)

    print("Downloading data from", url)
    urllib.request.urlretrieve(url, VALIDATION_ZIP)

    with zipfile.ZipFile(VALIDATION_ZIP, 'r') as zip_ref:
        zip_ref.extractall(dir)

    os.remove(VALIDATION_ZIP)


if __name__ == "__main__":
    download_validation_data()
    download_training_data()
