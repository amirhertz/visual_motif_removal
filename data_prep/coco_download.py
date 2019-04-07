from __future__ import print_function
from pycocotools.coco import COCO
import os
import zipfile
import shutil
import skimage.io as io
from PIL import Image
from threading import Thread, Lock
from urllib.error import URLError
from urllib.request import urlopen


# Choose main data dir and category from:
# person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop sign parking meter bench
# bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard
# sports ball kite baseball bat glove skateboard surfboard tennis racket bottle wine glass cup fork knife spoon
# bowl banana apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch potted plant bed dining table
# toilet tv laptop mouse remote keyboard cell phone microwave oven toaster sink refrigerator book clock vase scissors
# teddy bear hair drier toothbrush
data_dir = '../data/coco'
set_name = 'cat'
categories = ['cat']
download_count = 100000  # number of images to download
num_of_threads = 20

# consts
DATA_TYPE = 'val2014'
ANN_URL = 'http://images.cocodataset.org/annotations/annotations_train{}.zip'.format(DATA_TYPE)

# Setup paths
annDir = '%s/annotations' % data_dir
annZipFile = '%s/annotations_%s.zip' % (data_dir, DATA_TYPE)
annFile = '%s/instances_%s.json' % (annDir, DATA_TYPE)
images_dir = '%s/%s' % (data_dir, set_name)


def download_json():
    if not os.path.exists(annDir):
        os.makedirs(annDir)
    if not os.path.exists(annFile):
        if not os.path.exists(annZipFile):
            print("Downloading zipped annotations to " + annZipFile + " ...")
            resp = urlopen(ANN_URL)
            with open(annZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            resp.close()
        print("Unzipping " + annZipFile)
        with zipfile.ZipFile(annZipFile, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("... done unzipping")
    print("Will use annotations in " + annFile)


def download_setup():
    global download_count
    coco = COCO(annFile)
    img_cats = []
    img_ids = set()
    min_cat = download_count
    for category in categories:
        cat_ids = coco.getCatIds(catNms=[category])
        img_cats.append(coco.getImgIds(catIds=cat_ids))
        if len(img_cats[-1]) < min_cat:
            min_cat = len(img_cats[-1])
    for i in range(min_cat):
        for c in img_cats:
            if c[i] not in img_ids:
                img_ids.add(c[i])
    img_ids = list(img_ids)
    download_count = min(download_count, len(img_ids))
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return coco, img_ids


def download_thread(t, coco, img_ids):
    thread_count = download_count // num_of_threads + 1
    start_count = t * thread_count
    end_count = min(start_count + thread_count, download_count)
    for i in range(start_count, end_count):
        file_name = '%s/%04d.png' % (images_dir, i + 1)
        if not os.path.exists(file_name):
            for j in range(3):
                try:
                    lock.acquire()
                    img = coco.loadImgs(img_ids[i])[0]
                    lock.release()
                    img = io.imread(img['coco_url'])
                    img = Image.fromarray(img)
                    img.save(file_name, 'png')
                    break
                except URLError:
                    print('connection problem')
    print('work done: thread %d' % t)


def download_main():
    download_json()
    threads = []
    coco, img_ids = download_setup()
    for t in range(num_of_threads):
        threads.append(Thread(target=download_thread, args=(t, coco, img_ids)))
        threads[t].start()
    for t in range(num_of_threads):
        threads[t].join()
    print('all done')


if __name__ == '__main__':
    lock = Lock()
    download_main()
