import os
import shutil
import urllib.request
import zipfile

BASE_URL = "https://storage.googleapis.com/learning-datasets/"
TRAINING_DIR = "horse-or-human/training/"
TRAINING_ZIP = "horse-or-human.zip"
VALIDATION_ZIP = "validation-horse-or-human.zip"
VALIDATION_DIR = "horse-or-human/validation/"


def download_training_data():
    url = BASE_URL + TRAINING_ZIP

    if os.path.exists(TRAINING_DIR):
        shutil.rmtree(TRAINING_DIR)

    print("Downloading data from", url)
    urllib.request.urlretrieve(url, TRAINING_ZIP)

    with zipfile.ZipFile(TRAINING_ZIP, "r") as zip_ref:
        zip_ref.extractall(TRAINING_DIR)

    os.remove(TRAINING_ZIP)


def download_validation_data():
    url = BASE_URL + VALIDATION_ZIP

    if os.path.exists(VALIDATION_DIR):
        shutil.rmtree(VALIDATION_DIR)

    print("Downloading data from", url)
    urllib.request.urlretrieve(url, VALIDATION_ZIP)

    with zipfile.ZipFile(VALIDATION_ZIP, "r") as zip_ref:
        zip_ref.extractall(VALIDATION_DIR)

    os.remove(VALIDATION_ZIP)


if __name__ == "__main__":
    download_validation_data()
    download_training_data()
