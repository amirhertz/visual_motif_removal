from __future__ import print_function
import os
import zipfile
import shutil
from urllib.request import urlopen
from utils.train_utils import load_globals, init_folders
from utils.visualize_utils import run_net
import torch


# consts
DEVICE = torch.device('cpu')
ROOT_PATH = '.'
DATA_URL = 'http://pxcm.org/motif/demo.zip'
DATA_ZIP_FILE = '%s/demo.zip' % ROOT_PATH
NET_PATH = '%s/net_baseline.pth' % ROOT_PATH
TEST_PATH = '%s/test_images' % ROOT_PATH
RECONSTRUCTED_PATH = '%s/reconstructed_images' % ROOT_PATH


def download_data():
    if not os.path.exists(NET_PATH):
        if not os.path.exists(DATA_ZIP_FILE):
            print("Downloading zipped data to " + DATA_ZIP_FILE + " ...")
            resp = urlopen(DATA_URL)
            with open(DATA_ZIP_FILE, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            resp.close()
        print("Unzipping " + DATA_ZIP_FILE)
        with zipfile.ZipFile(DATA_ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall('./')
        print("... done unzipping")
    if os.path.exists(DATA_ZIP_FILE):
        os.remove(DATA_ZIP_FILE)
    print("Will use net_baseline in %s" % ROOT_PATH)


def run_demo():
    download_data()
    init_folders(RECONSTRUCTED_PATH)
    opt = load_globals(ROOT_PATH, {}, override=False)
    run_net(opt, DEVICE, ROOT_PATH, TEST_PATH, RECONSTRUCTED_PATH, 'demo')
    print("Reconstructed images are at %s" % RECONSTRUCTED_PATH)

if __name__ == '__main__':
    run_demo()
