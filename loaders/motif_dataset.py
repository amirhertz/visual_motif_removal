import math
import torch
from torch.utils.data import Dataset
from utils.image_utils import *
from utils.shapes_utils import *
import numpy as np
import os
import pickle
from PIL import Image
import random


class MotifDS(Dataset):
    def __init__(self, images_root, motifs_root, train=True, bound_offset=5, image_size=128,
                 motif_size=(40, 50), rgb=True, weight=(0.4, 0.6), perturbate=False, opacity_var=0.,
                 scale_vm=False, rotate_vm=False, crop_vm=False, batch_vm=0, font='', border=0, split_tag='split',
                 blur=False):
        super(MotifDS, self).__init__()
        self.__ds = ImageLoader(images_root)
        self.__size = image_size
        self.__w = weight
        self.__ov = opacity_var
        self.__train = train
        self.__perturbate = perturbate
        self.__rgb = rgb
        self.__scale_vm, self.__rotate, self.__crop, self.__batch_vm, = scale_vm, rotate_vm, crop_vm, batch_vm
        self.__vms = motif_size
        self.__offset = bound_offset
        self.__indices = self.__get_images_indices(images_root, train, os.path.basename(os.path.normpath(images_root)),
                                                   split_tag)
        self.__vm_paths, self.__is_text = None, None
        self.__blur = blur
        self.__vm_paths, self.__is_text = self.__get_motif_paths(motifs_root, split_tag)
        self.__counter = 0
        self.fonts = self.__get_fonts_paths(font)
        self.border = border

    def __getitem__(self, index):
        binary_mask, motifs, opacity_field = self.__generate_motif()
        or_image, sy_image = self.__generate_images(index, motifs, opacity_field)
        if self.__perturbate and random.random() < 0.5:
            sy_image = permute_image(sy_image, binary_mask, multiplier=random.randint(1, 2))
        or_image, sy_image, binary_mask, motifs = self.flip(or_image, sy_image, binary_mask, motifs)
        motifs, or_image, sy_image = self.trans(motifs, or_image, sy_image)
        motif_area = np.sum(binary_mask)
        if motif_area == 0:
            motif_area += 1
        return (torch.from_numpy(sy_image), torch.from_numpy(or_image), torch.from_numpy(binary_mask),
                torch.from_numpy(motifs))

    def __generate_motif(self):
        opacity_fields = []
        binary_mask = np.zeros([self.__size, self.__size, 1], dtype=np.float32)
        motif_rgb = np.zeros([self.__size, self.__size, 3], dtype=int)
        num_vm = 1 + self.__batch_vm // 2 + math.floor(random.random() * self.__batch_vm // 2)
        for vm_idx in range(num_vm):
            vm_rows, vm_cols, motif, vm_indices = [], None, None, None
            while type(vm_rows) is int or len(vm_rows) == 0:
                vm_size = self.__vms[0] + int(random.random() * (self.__vms[1] - self.__vms[0]))
                w = self.__w[0] + random.random() * (self.__w[1] - self.__w[0])
                if self.__vm_paths == 'shapes':
                    if random.random() < .5:
                        motif = generate_shape_motif(self.__rgb)
                        motif = distort_vm(motif, vm_size, scale=self.__scale_vm, crop=self.__crop,
                                           rotate=self.__rotate)
                    else:
                        motif = np.array(generate_line_motif(self.__rgb, self.__size))

                else:
                    vm_index = random.randint(0, len(self.__vm_paths) - 1)

                    if self.__is_text:
                        motif = self.__generate_text_motif(self.__vm_paths[vm_index])
                    else:
                        motif = self.__vm_paths[vm_index]

                    motif = distort_vm(motif, vm_size, scale=self.__scale_vm, crop=self.__crop,
                                       rotate=self.__rotate, gray=self.__rgb == 'gray' and not self.__is_text)
                if motif is not False:
                    vm_indices, vm_rows, vm_cols = get_image_indices(motif)
                    motif[vm_indices[0], vm_indices[1], :3] = 255
                    if vm_cols < self.__size:
                        offset_rows = random.randint(0, self.__size - vm_rows - 1)
                    else:
                        offset_rows = [0]
                    if vm_rows < self.__size:
                        offset_cols = random.randint(0, self.__size - vm_cols - 1)
                    else:
                        offset_cols = 0
                    vm_rows, vm_cols = offset_rows + vm_indices[0], offset_cols + vm_indices[1]
                else:
                    vm_rows = 0

            motif_rgb[vm_rows, vm_cols, :] = motif[vm_indices[0], vm_indices[1], :-1]
            field = np.zeros([self.__size, self.__size, 1], dtype=np.float32)
            field[vm_rows, vm_cols, 0] = 1
            vm_rows, vm_cols, _ = np.nonzero(field)
            field[vm_rows, vm_cols, 0] = get_opacity_field(self.__size, w, self.__ov)[vm_rows, vm_cols]
            if self.__blur and random.random() < .5:
                k_size = random.randint(1, 2)
                blurred = cv2.blur(field, (1 + k_size * 2, 1 + k_size * 2))
                field[vm_rows, vm_cols, 0] = blurred[vm_rows, vm_cols]
            binary_mask[vm_rows, vm_cols, 0] = 1
            opacity_fields.append(field)
        return binary_mask, motif_rgb, opacity_fields

    def __generate_text_motif(self, text):
        color = (255, 255, 255, 255)
        font = random.choice(self.fonts)
        if self.__rgb == 'gray':
            color = random.randint(180, 255)
            color = (color, color, color, 255)
        elif type(self.__rgb) is tuple:
            color = list(self.__rgb)
        elif self.__rgb:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        return get_text_motif(text, color, font=font, border=self.border)

    def __generate_images(self, index, motif, fields):
        image_index = self.__indices[index]
        image, _ = self.__ds[image_index]
        image = np.array(crop_image(image, self.__size, rand=True))
        if len(image.shape) != 3:
            image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
        if motif is None:
            sy_image = None
        else:
            sy_image = image
            for op in fields:
                opacity = np.repeat(op, 3, axis=2)
                sy_image = (1 - opacity) * sy_image + opacity * motif
        return image, sy_image

    def __len__(self):
        return len(self.__indices)

    def __get_images_indices(self, root, train, domain, split_tag):

        indices_path = '%s/images_%s.pkl' % (root, split_tag)
        if os.path.isfile(indices_path):
            with open(indices_path, 'rb') as f:
                indices_main = pickle.load(f)
        else:
            indices_main = dict()
        if domain not in indices_main:
            total = len(self.__ds)
            indices = np.arange(total)
            if 'coco' in root:
                random.shuffle(indices)
                indices_main[domain] = np.split(indices, [int(0.1 * total)])
                with open(indices_path, 'wb') as f:
                    pickle.dump(indices_main, f, pickle.HIGHEST_PROTOCOL)
            else:
                return indices
        return indices_main[domain][int(train)]

    def __get_motif_paths(self, root, split_tag):
        if root == 'shapes':
            return 'shapes', False
        if os.path.isfile(root):
            filename, file_extension = os.path.splitext(root)
            if file_extension == '.txt':
                return self.__extract_word(root), True
            else:
                return [root], False
        if not os.path.isdir(root):
            raise ValueError(f'Watermarks root: {root} doesn\'t exist')
        data_split_path = '%s/vm_%s.pkl' % (root, split_tag)
        split_dict = {'train': None, 'test': None}
        if os.path.isfile(data_split_path):
            with open(data_split_path, 'rb') as f:
                split_dict = pickle.load(f)
        else:
            paths = []
            for root, _, files in os.walk(root):
                for file in files:
                    file_name, file_extension = os.path.splitext(file)
                    if file_extension == '.png':
                        paths.append(os.path.join(root, file))
            paths = sorted(paths)
            random.shuffle(paths)
            split_place = min(120, int(len(paths) * 0.9))
            split_dict['train'] = paths[split_place:]
            split_dict['test'] = paths[:split_place]
            with open(data_split_path, 'wb') as f:
                pickle.dump(split_dict, f, pickle.HIGHEST_PROTOCOL)
        if self.__train:
            return split_dict['train'], False
        else:
            return split_dict['test'], False

    @staticmethod
    def __get_fonts_paths(root):
        if os.path.isfile(root):
            return [root]
        if not os.path.isdir(root):
            return ''
        font_list_path = '%s/font_list.pkl' % root
        if os.path.isfile(font_list_path):
            with open(font_list_path, 'rb') as f:
                paths = pickle.load(f)['paths']
        else:
            paths = []
            for root, _, files in os.walk(root):
                for file in files:
                    file_name, file_extension = os.path.splitext(file)
                    if file_extension == '.ttf' or file_extension == '.otf':
                        paths.append(os.path.join(root, file))
            with open(font_list_path, 'wb') as f:
                pickle.dump({'paths': paths}, f, pickle.HIGHEST_PROTOCOL)
        return paths

    def __extract_word(self, root):
        file = open(root, 'r')
        raw_text = file.read().split(' ')
        file.close()
        if len(raw_text) == 1:
            return raw_text
        if self.__train:
            return raw_text[:int(len(raw_text) * 0.9)]
        else:
            return raw_text[int(len(raw_text) * 0.9):]

    @staticmethod
    def trans(*images):
        transformed = []
        for image in images:
            transformed.append((image / 127.5 - 1).astype(np.float32))
        return transformed

    @staticmethod
    def flip(*images):
        flipped = []
        for image in images:
            flipped.append(np.transpose(image, (2, 0, 1)))
        return flipped


class ImageLoader:
    def __init__(self, root):
        self.__paths = self.__init_paths(root)

    def __getitem__(self, index):
        return self.__load_image(index)

    def __len__(self):
        return len(self.__paths)

    def __load_image(self, index):
        image = Image.open(self.__paths[index])
        
        # Make sure the image is RGB, because some files are loaded as CMYK
        if image.mode == 'CMYK':
            image = image.convert('RGB')
        
        return image, 0

    @staticmethod
    def __init_paths(root):
        if not os.path.isdir(root):
            raise ValueError("Image root doesn't exist: %s" % root)
        paths = []
        for root, _, files in os.walk(root):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if file_extension == '.png' or file_extension == '.jpg':
                    paths.append(os.path.join(root, file))
        paths = sorted(paths)
        return paths
