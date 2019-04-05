import torch
from torch.utils.data import Dataset
from utils.image_utils import *
import numpy as np
import os
from PIL import Image


class CacheLoader(Dataset):

    def __init__(self, images_root, train=True, patch_size=None):
        super(CacheLoader, self).__init__()
        self.patch_size = patch_size
        self.roots = self.init_root(images_root, train)
        self.extension = ['png' for _ in self.roots]
        self.sizes = self.cal_sizes()
        self.size = sum(self.sizes)
        print('%s dataset size: %d' %( 'train' if train else 'test', self.size))

    @staticmethod
    def init_root(images_root, train):
        if train:
            sub = 'train'
        else:
            sub = 'test'
        if type(images_root) is not list:
            images_root = [images_root]
        return ['%s/%s' % (root, sub) for root in images_root]

    def cal_sizes (self):
        sizes = []
        for root_id, images_root in enumerate(self.roots):
            size = 0
            for root, _, files in os.walk(images_root):
                for file in files:
                    file_name, file_extension = os.path.splitext(file)
                    if file_extension == '.jpg':
                        self.extension[root_id] = 'jpg'
                    if (file_extension == '.png' or file_extension == '.jpg') and 'bounding' not in file_name and 'opc' not in file_name:
                        size += 1
            sizes.append(size // 4)
        return sizes

    def __getitem__(self, index):
        main_path, root_id = self.get_path(index)
        or_image = self.load_image('%s_real_image.%s' % (main_path, self.extension[root_id]))
        sy_image = self.load_image('%s_synthesized.%s' % (main_path, self.extension[root_id]))
        binary_mask = self.load_image('%s_real_mask.%s' % (main_path, self.extension[root_id]), True)
        motif_rgb = self.load_image('%s_real_motif.%s' % (main_path, self.extension[root_id]))
        if self.patch_size:
            sy_image, or_image, binary_mask, motif_rgb = \
                self.crop_images(sy_image, or_image, binary_mask, motif_rgb)
        motif_area = np.sum(binary_mask)
        if motif_area == 0:
            motif_area += 1
        motif_area = motif_area.astype(np.float32)
        return (torch.from_numpy(sy_image), torch.from_numpy(or_image), torch.from_numpy(binary_mask),
                torch.from_numpy(motif_rgb),motif_area)

    def get_path(self, index):
        root_id = 0
        for idx, size in enumerate(self.sizes):
            if index >= size:
                index -= size
                root_id += 1
            else:
                return '%s/%d' % (self.roots[idx], index), root_id

    def crop_images(self, *images):
        size = images[0].shape[1]
        left_most = random.randint(0, size - 1 - self.patch_size)
        top_most = random.randint(0, size - 1 - self.patch_size)
        cropped = []
        for image in images:
            cropped.append(image[:, left_most: left_most + self.patch_size, top_most: top_most + self.patch_size])
        return cropped

    @staticmethod
    def load_image(path, gray=False):
        image = np.array(Image.open(path))
        if gray:
            image = (image / 255).astype(np.float32)
            if len(image.shape) == 3:
                image = image[:, :, 0]
            image = np.expand_dims(image, 0)
        else:
            image = (image / 127.5 - 1).astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
        return image

    def __len__(self):
        return self.size

