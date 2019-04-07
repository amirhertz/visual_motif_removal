from torch.utils.data import DataLoader
from utils.train_utils import load_globals, init_folders
from loaders.motif_dataset import MotifDS
from PIL import Image
import numpy as np


# dataset configurations
dataset_tag = 'demo_text_vm_ds'
images_root = 'path to images'
cache_root = 'path  to data dir/%s' % dataset_tag
vm_root = '../data/text/word2.txt'
image_size = 512
vm_size = (50, 150)
rotate_vm = True
scale_vm = True
crop_vm = True
batch_vm = 13
weight = (0.2, 0.8)
use_rgb = False
perturbate = False
opacity_var = 0.
font = 'path to fonts dir'
text_border = 3
blur = True

# number of images, type
num_train = 5000
num_test = 500
save_extension = 'jpg'


def init_loaders(opt):
    train_dataset = MotifDS(opt.images_root, opt.vm_root, train=True, image_size=opt.image_size,
                            motif_size=opt.vm_size, weight=opt.weight, perturbate=opt.perturbate,
                            opacity_var=opt.opacity_var, rgb=opt.use_rgb, scale_vm=opt.scale_vm,
                            rotate_vm=opt.rotate_vm, crop_vm=opt.crop_vm, batch_vm=opt.batch_vm, font=opt.font,
                            border=opt.text_border, split_tag=dataset_tag, blur=opt.blur)

    test_dataset = MotifDS(images_root, vm_root, train=False, image_size=opt.image_size, motif_size=opt.vm_size,
                           weight=opt.weight, perturbate=opt.perturbate, opacity_var=opt.opacity_var, rgb=opt.use_rgb,
                           scale_vm=False, rotate_vm=opt.rotate_vm, crop_vm=False, batch_vm=opt.batch_vm,
                           font=opt.font, border=opt.text_border, split_tag=dataset_tag, blur=opt.blur)

    _train_data_loader = DataLoader(train_dataset, batch_size=524288 // (image_size ** 2), shuffle=True, num_workers=2)
    _test_data_loader = DataLoader(test_dataset, batch_size=524288 // (image_size ** 2), shuffle=True, num_workers=2)
    return _train_data_loader, _test_data_loader


def transform_to_numpy_image(tensor_image):
    image = tensor_image.cpu().detach().numpy()
    image = np.transpose(image, (0, 2, 3, 1))
    if image.shape[3] != 3:
        image = np.squeeze(image, axis=3)
        # image = np.repeat(image, 3, axis=3)
    else:
        image = (image / 2 + 0.5)
    return image


def save_np_image(images, folder, suffix, start_count=0):
    images = (images * 255).astype(np.uint8)
    for image_index in range(images.shape[0]):
        image_path = '%s/%d_%s.%s' % (folder, image_index + start_count, suffix, save_extension)
        image = Image.fromarray(images[image_index])
        image.save(image_path)


def save_dataset(folder, num_elem, loader):
    counter = 0
    image_suffixes = ['synthesized', 'real_image', 'real_mask', 'real_motif']
    while counter < num_elem:
        for data in loader:
            if counter + data[0].shape[0] > num_elem:
                for i in range(len(data)):
                    data[i] = data[i][0: num_elem - counter]
            numpy_images = []

            for i in range(len(data)):
                numpy_images.append(transform_to_numpy_image(data[i]))
            for i in range(len(image_suffixes)):
                save_np_image(numpy_images[i], folder, image_suffixes[i], start_count=counter)
            counter += data[0].shape[0]
            if counter >= num_elem:
                break
        print(counter)


def run_cache():
    train_root = '%s/train' % cache_root
    test_root = '%s/test' % cache_root
    init_folders(train_root, test_root)
    _opt = load_globals(cache_root, globals(), override=True)
    _train_data_loader, _test_data_loader = init_loaders(_opt)
    save_dataset(train_root, num_train, _train_data_loader)
    save_dataset(test_root, num_test, _test_data_loader)


if __name__ == '__main__':
    run_cache()
