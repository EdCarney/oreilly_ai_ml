import urllib.request as req
import zipfile
import os
import sys


DATASET_MAP = {
        "TWITTER_2B": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "COMMON_CRAWL_32B": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
        "COMMON_CRAWL_840B": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
        "GIGAWORD_6B": "https://nlp.stanford.edu/data/glove.6B.zip"
        }


# function to download zip file from URL and extract resulting file into a dir
# note that outfile name must end in a .zip extension
def download_and_extract_file(url: str, outfile: str = None) -> None:
    if outfile is None:
        outfile = url[url.rfind("/") + 1:]
    elif not outfile.endswith(".zip"):
        print("ERROR: outfile path must end in a .zip extension")
        exit(1)

    extract_dir = outfile[:-4]

    if os.path.exists(extract_dir) or os.path.exists(outfile):
        print("ERROR: outfile or extractdir paths already exist")
        exit(1)

    print("Retrieving data from: ", url)

    req.urlretrieve(url, outfile)

    print("Extracting data to: ", extract_dir)

    zip_ref = zipfile.ZipFile(outfile, mode='r')
    zip_ref.extractall(extract_dir)
    os.remove(outfile)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("ERROR: must minimally specify glove dataset")
        exit(1)

    dataset_name = sys.argv[1].upper()
    outfile_name = sys.argv[2] if len(sys.argv) > 2 else None

    if dataset_name not in DATASET_MAP.keys():
        print("ERROR: dataset name not known, must be one of the following:")
        for name in DATASET_MAP.keys():
            print("\t", name)
        exit(1)

    download_and_extract_file(DATASET_MAP[dataset_name], outfile_name)
